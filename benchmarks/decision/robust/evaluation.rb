module DecisionRobust
  module_function

  def collect(model, tokenizer, raw, requested)
    rows = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    logits = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: requested, batch_size: 4).collect(rows)
    EasyAI::Runtime::DevicePolicy.new(requested: requested, budget_mib: 4096).check_budget!(requested)
    [rows, logits]
  rescue Torch::Error, EasyAI::Runtime::DevicePolicy::MemoryBudgetExceeded => error
    policy = EasyAI::Runtime::DevicePolicy.new(requested: requested, budget_mib: 4096)
    raise unless requested == "cuda" && policy.recoverable?(error)
    model.to("cpu")
    GC.start
    puts "Evaluation falls back to CPU: #{error.message.lines.first}"
    collect(model, tokenizer, raw, "cpu")
  end

  def measure(model, tokenizer, raw, requested, calibrator: EasyAI::Decision::Calibrator.new, predictions_path: nil)
    if predictions_path
      batches = raw.each_slice(1000).to_a
      # Keep complete fact pairs together when reporting: collect chunks, summarize the full panel.
      rows, logits = [], []
      batches.each_with_index do |batch, index|
        examples, values = collect(model, tokenizer, batch, requested)
        rows.concat(examples)
        logits.concat(values)
        puts "#{File.basename(predictions_path)} #{[1000 * (index + 1), raw.size].min}/#{raw.size}"
      end
    else
      rows, logits = collect(model, tokenizer, raw, requested)
    end
    predictions = raw.each_with_index.map do |row, index|
      row.merge("logits" => logits[index], "probabilities" => calibrator.probabilities(logits[index]),
        "prediction" => rows[index].options.fetch(logits[index].each_index.max_by { |j| logits[index][j] }).fetch("id"))
    end
    write_rows(predictions_path, predictions) if predictions_path
    metrics = DecisionFactual.summarize(raw, rows, logits, calibrator)
    metrics["raw"] = DecisionFactual.summarize(raw, rows, logits, EasyAI::Decision::Calibrator.new) if predictions_path
    metrics.merge!(robust_metrics(predictions))
    metrics
  end

  def robust_metrics(predictions)
    raw = predictions.select { |row| row.dig("world", "version") == 3 }
    axes = %w[actor_switch question_flip fact_flip order wording irrelevant].to_h do |axis|
      cells = raw.select { |row| row.fetch("contrast_groups").key?(axis) }.group_by { |row| row.fetch("language") }.transform_values do |items|
        groups = items.group_by { |row| row.fetch("contrast_groups").fetch(axis) }.values
        raise "Incomplete #{axis} groups" unless groups.all? { |group| group.size == 2 }
        result = { "groups" => groups.size, "all_correct" => groups.count { |group| group.all? { |row| row.fetch("prediction") == row.fetch("target") } }.fdiv(groups.size) }
        if axis == "actor_switch"
          mixed, same = groups.partition { |group| group.first.fetch("world").fetch("truth_pattern").uniq.size == 2 }
          result["mixed_all_correct"] = mixed.empty? ? nil : mixed.count { |group| group.all? { |row| row.fetch("prediction") == row.fetch("target") } }.fdiv(mixed.size)
          result["same_all_correct"] = same.empty? ? nil : same.count { |group| group.all? { |row| row.fetch("prediction") == row.fetch("target") } }.fdiv(same.size)
          result["mixed_groups"], result["same_groups"] = mixed.size, same.size
        end
        result
      end
      [axis, cells]
    end
    languages = raw.group_by { |row| row.fetch("language") }.transform_values do |items|
      known, unknown = items.partition { |row| row.fetch("target") != "unknown" }
      words = items.select { |row| row.fetch("world").fetch("variant") == "wording" }
      worlds = items.group_by { |row| row.fetch("group_id") }.values
      by_slice = items.group_by { |row| row.fetch("world").fetch("phenomenon") }.transform_values { |rows| accuracy(rows) }
      by_pattern = items.group_by { |row| row.fetch("world").fetch("truth_pattern").map { |value| value ? 1 : 0 }.join }.transform_values { |rows| accuracy(rows) }
      { "count" => items.size, "known_accuracy" => accuracy(known), "unknown_accuracy" => accuracy(unknown),
        "wording_accuracy" => accuracy(words), "world_all_correct" => worlds.count { |group| group.all? { |row| row.fetch("prediction") == row.fetch("target") } }.fdiv(worlds.size),
        "world_count" => worlds.size, "by_phenomenon" => by_slice, "by_truth_pattern" => by_pattern,
        "unknown_confusion" => unknown.group_by { |row| row.fetch("prediction") }.transform_values(&:size) }
    end
    languages.each do |language, value|
      cell = axes.fetch("actor_switch").fetch(language)
      value["binding_all_correct"] = cell.fetch("all_correct")
      value["mixed_binding_all_correct"] = cell.fetch("mixed_all_correct")
      value["same_binding_all_correct"] = cell.fetch("same_all_correct")
    end
    { "metrics_version" => 2, "robust_by_language" => languages, "contrast_axes" => axes }
  end

  def accuracy(rows)
    rows.empty? ? nil : rows.count { |row| row.fetch("prediction") == row.fetch("target") }.fdiv(rows.size)
  end

  def final_candidates(root)
    protocol = verify(root)
    decision = JSON.parse(File.read(File.join(root, "confirmation.json")))
    names = ARMS.map { |arm| "#{arm}-1337" }
    names << "candidate-#{protocol.fetch('confirmation').fetch('seed')}" if decision.fetch("confirmation_required")
    names.to_h do |name|
      result = JSON.parse(File.read(File.join(root, name, "summary.json")))
      raise "Pilot/confirmation incomplete" unless result.fetch("step") == STEPS
      [name, result.fetch("selected")&.fetch("checkpoint") || result.fetch("last_checkpoint")]
    end.merge("release" => PARENT)
  end

  def evaluate(root)
    candidates = final_candidates(root)
    marker = File.join(root, "acceptance-opened.json")
    raise "Acceptance already opened; no test-driven retuning" if File.exist?(marker)
    calibration = raw_rows(File.join(root, "data/calibration.jsonl"))
    policies = candidates.to_h do |name, path|
      loaded = EasyAI::Decision::Checkpoint.load(path)
      rows, logits = collect(loaded.fetch(:model), loaded.fetch(:tokenizer), calibration, device)
      temperature = EasyAI::Decision::Calibrator.new.fit(logits, rows.map(&:target_index)).temperature
      policy = { "checkpoint" => loaded.fetch(:path), "weights_sha256" => loaded.fetch(:weights_fingerprint), "temperature" => temperature }
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      [name, policy]
    end
    write(File.join(root, "calibration.json"), policies)
    write(marker, { "opened_at" => Time.now.utc.iso8601, "protocol_sha256" => sha(File.join(root, "protocol.json")),
      "policies_sha256" => sha(File.join(root, "calibration.json")), "confirmation_sha256" => sha(File.join(root, "confirmation.json")),
      "scope" => "Checkpoint selection, confirmation and calibration frozen before all acceptance predictions" })
    raw = raw_rows(File.join(root, "data/test.jsonl"))
    regression = raw_rows(File.join(root, "data/regression.jsonl"))
    results = policies.to_h do |name, policy|
      loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
      raise "Calibrated weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
      calibrator = EasyAI::Decision::Calibrator.new(temperature: policy.fetch("temperature"))
      value = measure(loaded.fetch(:model), loaded.fetch(:tokenizer), raw, device, calibrator: calibrator,
        predictions_path: File.join(root, "predictions-#{name}.jsonl"))
      value["regression"] = DecisionFactual.measure(loaded.fetch(:model), loaded.fetch(:tokenizer), regression, device, calibrator: calibrator,
        predictions_path: File.join(root, "regression-#{name}.jsonl"))
      write(File.join(root, "evaluation-#{name}.json"), value)
      puts "#{name} final macro=#{value['factual_macro_accuracy']} cells=#{value['robust_by_language']}"
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      [name, value]
    end
    write(File.join(root, "evaluation.json"), results)
  end

  def verify_opened(root)
    protocol = verify(root)
    marker = JSON.parse(File.read(File.join(root, "acceptance-opened.json")))
    { "protocol.json" => "protocol_sha256", "calibration.json" => "policies_sha256", "confirmation.json" => "confirmation_sha256" }.each do |file, key|
      raise "Frozen #{file} changed after acceptance" unless sha(File.join(root, file)) == marker.fetch(key)
    end
    protocol
  end
end
