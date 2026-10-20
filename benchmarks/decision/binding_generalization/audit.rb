module DecisionBindingGeneralization
  module_function

  def audit_early(root, variant, seed, step, device: "cpu")
    verify(root)
    device = device.to_s
    raise ArgumentError, "Early audit is the fixed CE learning-curve reference" unless variant.to_s == "ce" && [1337, 2027].include?(seed) && step == 1500 && %w[cpu cuda].include?(device)
    paths = Dir[File.join(root, "arms", "ce-#{seed}", "choice/checkpoints/step-#{format('%08d', step)}-*")].select { |path| File.file?(File.join(path, "manifest.json")) }
    raise "Early checkpoint missing or ambiguous" unless paths.any? && paths.map { |path| sha(File.join(path, "weights.pt")) }.uniq.size == 1
    loaded = EasyAI::Decision::Checkpoint.load(paths.first)
    raise "Wrong early checkpoint step" unless loaded.fetch(:metadata).fetch("training").fetch("step") == step
    metrics = %w[probe canonical seen_combinations novel_actor novel_event].to_h do |name|
      value = DecisionScaling.measure(loaded.fetch(:model), loaded.fetch(:tokenizer), raw(root, name), device)
      raise "Early audit fell back" unless loaded.fetch(:model).parameters.all? { |parameter| parameter.device.to_s.start_with?(device) }
      [name, value]
    end
    value = { "variant" => "ce", "seed" => seed, "step" => step, "device" => device, "weights_sha256" => loaded.fetch(:weights_fingerprint),
      "training_probe" => metrics.fetch("probe"), "development" => metrics.except("probe"),
      "scope" => "Post-training read-only evaluation of the retained 1500-step snapshot. Same dataset, seed and trajectory as 6000; no new optimization, selection or acceptance. Used to distinguish training duration from the historical cross-data comparison." }
    directory = File.join(root, "audit", "ce-#{seed}-#{step}")
    FileUtils.mkdir_p(directory)
    write(File.join(directory, "#{device}.json"), value)
    puts "Audited early CE/#{seed}/#{step} on #{device}"
    value
  end

  def audit_endpoint(root, variant, seed, steps, device: "cpu")
    verify(root)
    device = device.to_s
    raise ArgumentError, "Audit device must be explicit" unless %w[cpu cuda].include?(device)
    raise ArgumentError, "Unknown audit endpoint" unless VARIANTS.include?(variant.to_s) && [1337, 2027].include?(seed) && [6000, 12000].include?(steps)
    result_path = File.join(root, "arms", "#{variant}-#{seed}", "result-#{steps}.json")
    result = JSON.parse(File.read(result_path))
    loaded = EasyAI::Decision::Checkpoint.load(result.fetch("checkpoint"))
    directory = File.join(root, "audit", "#{variant}-#{seed}-#{steps}")
    FileUtils.mkdir_p(directory)
    canonical_predictions = {}
    views = %w[canonical seen_combinations novel_actor].to_h do |name|
      rows = raw(root, name)
      _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, device, gc_interval: 8)
      raise "Audit silently fell back" unless loaded.fetch(:model).parameters.all? { |parameter| parameter.device.to_s.start_with?(device) }
      predictions = rows.each_index.map { |index| rows[index].fetch("options").fetch(logits[index].each_index.max_by { |position| logits[index][position] }).fetch("id") }
      canonical_predictions = rows.each_index.to_h { |index| [rows[index].fetch("id"), predictions[index]] } if name == "canonical"
      official = DecisionJudgment.metrics(rows, logits)
      independent = DecisionScaling.independent_actor_pairs(rows, predictions)
      independent.each do |language, cell|
        raise "v07 actor count mismatch" unless cell.fetch("groups") == official.dig("by_language", language, "actor_binding_groups")
        raise "v07 actor correctness mismatch" unless cell.fetch("all_correct") == official.dig("by_language", language, "actor_binding_all_correct")
      end
      path = File.join(directory, "#{device}-#{name}.jsonl")
      DecisionRobust.write_rows(path, rows.each_index.map { |index| rows[index].merge("prediction" => predictions[index], "logits" => logits[index]) })
      compact = DecisionScaling.compact_metrics(official)
      recorded = DecisionScaling.compact_metrics(result.fetch("development").fetch(name))
      matches = compact.all? do |language, metrics|
        %w[accuracy balanced actor_binding_all_correct event_binding_all_correct].all? { |key| metrics.fetch(key) == recorded.fetch(language).fetch(key) }
      end
      [name, { "metrics" => compact, "independent_actor_pairs" => independent, "matches_recorded_classification_metrics" => matches,
        "predictions_sha256" => sha(path) }]
    end
    loaded.fetch(:model).to("cpu")
    predictor = EasyAI::Decision::Predictor.load(result.fetch("checkpoint"), device: device, candidate_chunk_size: 1)
    raise "Public audit fell back" unless predictor.device == device
    samples = raw(root, "canonical").select { |row| row.dig("world", "mixed") && row.fetch("source").end_with?("Actor") }
      .group_by { |row| row.fetch("language") }.values.flat_map { |rows| rows.first(4) }
    checks = samples.map do |row|
      original = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options"))
      reversed = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options").reverse)
      probabilities = original.fetch("probabilities")
      delta = probabilities.keys.map { |key| (probabilities.fetch(key) - reversed.fetch("probabilities").fetch(key)).abs }.max
      raise "Public v07 output invalid" unless probabilities.values.all?(&:finite?) && (probabilities.values.sum - 1).abs < 1e-6 && delta < 1e-4
      guess = probabilities.max_by { |_, probability| probability }.first
      { "id" => row.fetch("id"), "language" => row.fetch("language"), "sum" => probabilities.values.sum, "permutation_delta" => delta,
        "prediction" => guess, "batch_prediction" => canonical_predictions.fetch(row.fetch("id")), "batch_public_prediction_agrees" => guess == canonical_predictions.fetch(row.fetch("id")) }
    end
    value = { "device" => device, "result_sha256" => sha(result_path), "weights_sha256" => sha(File.join(result.fetch("checkpoint"), "weights.pt")),
      "views" => views, "public_interface" => checks,
      "scope" => "Observed development only, no updates. Independent pair counts, recorded-classification comparison, raw predictions and public normalized/permuted JSON probabilities. CPU/GPU label differences, if any, remain explicit." }
    write(File.join(directory, "#{device}.json"), value)
    puts "Audited #{variant}/#{seed}/#{steps} on #{device}"
    value
  end
end
