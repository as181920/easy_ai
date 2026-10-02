module DecisionFactual
  module_function

  def measure(model, tokenizer, raw, device, calibrator: EasyAI::Decision::Calibrator.new, predictions_path: nil, include_raw: false)
    rows = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: device, batch_size: 4)
    logits = if predictions_path
      rows.each_slice(1000).with_index.flat_map do |chunk, index|
        values = evaluator.collect(chunk)
        puts "Evaluation #{File.basename(predictions_path)}: #{[(index + 1) * 1000, rows.size].min}/#{rows.size}"
        values
      end
             else
      evaluator.collect(rows)
             end
    if predictions_path
      predictions = raw.each_with_index.map do |row, index|
        row.merge("logits" => logits[index], "probabilities" => calibrator.probabilities(logits[index]),
          "prediction" => rows[index].options.fetch(logits[index].each_index.max_by { |j| logits[index][j] }).fetch("id"))
      end
      write_rows(predictions_path, predictions)
    end
    metrics = summarize(raw, rows, logits, calibrator)
    metrics["raw"] = summarize(raw, rows, logits, EasyAI::Decision::Calibrator.new) if include_raw
    metrics
  end

  def summarize(raw, rows, logits, calibrator)
    correct = rows.each_index.map { |index| logits[index].each_index.max_by { |j| logits[index][j] } == rows[index].target_index }
    metrics = EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index), calibrator)
    cells = rows.each_index.group_by { |i| [rows[i].source, rows[i].language] }.transform_values do |indexes|
      EasyAI::Decision::Evaluator.metrics(indexes.map { |i| logits[i] }, indexes.map { |i| rows[i].target_index }, calibrator)
    end
    factual = cells.reject { |(source, _), _| source == "MASSIVE-Scenario" }
    language_cells = factual.group_by { |(_, language), _| language }.values
    metrics["factual_macro_accuracy"] = language_cells.sum { |items| items.sum { |_, cell| cell.fetch("accuracy") }.fdiv(items.size) }.fdiv(language_cells.size)
    factual_indexes = rows.each_index.reject { |i| rows[i].source == "MASSIVE-Scenario" }
    metrics["factual_nll"] = calibrator.nll(factual_indexes.map { |i| logits[i] }, factual_indexes.map { |i| rows[i].target_index })
    metrics["factual"] = EasyAI::Decision::Evaluator.metrics(factual_indexes.map { |i| logits[i] },
      factual_indexes.map { |i| rows[i].target_index }, calibrator)
    metrics["routing_by_language"] = cells.filter_map { |(source, language), cell| [language, cell] if source == "MASSIVE-Scenario" }.to_h
    metrics["by_source_language"] = cells.to_h { |(source, language), cell| ["#{source}/#{language}", cell] }
    pairs = rows.each_index.select { |i| rows[i].contrast_groups["fact_flip"] }
      .group_by { |i| [rows[i].language, rows[i].contrast_groups.fetch("fact_flip")] }
    raise "Incomplete factual evaluation pairs" unless pairs.values.all? { |indexes| indexes.size == 2 }
    metrics["pair_all_correct"] = pairs.empty? ? nil : pairs.values.count { |indexes| indexes.all? { |i| correct[i] } }.fdiv(pairs.size)
    metrics["pairs_by_language"] = pairs.group_by { |(language, _), _| language }.transform_values do |items|
      { "count" => items.size, "all_correct" => items.count { |_, indexes| indexes.all? { |i| correct[i] } }.fdiv(items.size) }
    end
    metrics["factual_by_language"] = factual.group_by { |(_, language), _| language }.transform_values do |items|
      { "source_macro_accuracy" => items.sum { |_, cell| cell.fetch("accuracy") }.fdiv(items.size) }
    end
    metrics["natural_by_language"] = cells.reject { |(source, _), _| source.start_with?("Factual-") || source == "MASSIVE-Scenario" }
      .group_by { |(_, language), _| language }.transform_values do |items|
        { "source_macro_accuracy" => items.sum { |_, cell| cell.fetch("accuracy") }.fdiv(items.size) }
      end
    metrics["by_phenomenon"] = raw.each_index.select { |i| raw[i]["world"] }.group_by { |i| raw[i].fetch("world").fetch("phenomenon") }
      .transform_values { |indexes| { "count" => indexes.size, "accuracy" => indexes.count { |i| correct[i] }.fdiv(indexes.size) } }
    metrics["by_domain_language"] = raw.each_index.select { |i| raw[i]["world"] }
      .group_by { |i| [raw[i].fetch("world").fetch("event"), rows[i].language] }.to_h do |(event, language), indexes|
        ["#{event}/#{language}", EasyAI::Decision::Evaluator.metrics(indexes.map { |i| logits[i] }, indexes.map { |i| rows[i].target_index }, calibrator)]
      end
    metrics["by_target_language"] = rows.each_index.group_by { |i| [rows[i].target, rows[i].language] }.to_h do |(target, language), indexes|
      ["#{target}/#{language}", { "count" => indexes.size, "accuracy" => indexes.count { |i| correct[i] }.fdiv(indexes.size) }]
    end
    metrics["by_language"] = rows.each_index.group_by { |i| rows[i].language }.transform_values do |indexes|
      EasyAI::Decision::Evaluator.metrics(indexes.map { |i| logits[i] }, indexes.map { |i| rows[i].target_index }, calibrator)
    end
    metrics
  end

  def diagnostics(model, tokenizer, raw, device)
    rows = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: device, batch_size: 4)
    original = evaluator.collect(rows)
    permutations = evaluator.collect(rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) })
    perturbations = { "state_removed" => rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("state" => "[context withheld]")) },
      "question_removed" => rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("question" => "[question withheld]")) } }
    rotated = rows.each_index.group_by { |i| rows[i].language }.values.flat_map do |indexes|
      indexes.each_with_index.map do |index, offset|
        [index, EasyAI::Decision::Data::Example.new(rows[index].to_h.merge("state" => rows[indexes[(offset + 1) % indexes.size]].state))]
      end
    end.sort_by(&:first).map(&:last)
    perturbations["shuffled_state"] = rotated
    perturbations["candidate_wording"] = rows.map do |row|
      labels = row.language == "zh-CN" ? { "yes" => "与记录一致", "no" => "与记录矛盾", "unknown" => "证据不足" } :
        { "yes" => "supported by the record", "no" => "contradicted by the record", "unknown" => "insufficient information" }
      options = row.options.map { |option| option.merge("text" => labels.fetch(option.fetch("id"))) }
      EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => options))
    end
    result = perturbations.transform_values do |changed|
      logits = evaluator.collect(changed)
      { "accuracy_with_original_targets" => EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index)).fetch("accuracy"),
        "mean_max_logit_change" => original.zip(logits).sum { |a, b| a.zip(b).map { |x, y| (x - y).abs }.max }.fdiv(rows.size) }
    end
    result["permutation_max_logit_error"] = original.zip(permutations).flat_map { |a, b| a.zip(b.reverse).map { |x, y| (x - y).abs } }.max
    result["unique_state_token_sequences"] = rows.map { |row| tokenizer.encode(row.state) }.uniq.size
    result["scope"] = "Removed fields retain original targets for sensitivity only, not new gold labels. All inputs are training diagnostics."
    result
  end
end

module DecisionFactual
  module_function

  def confirm(root)
    protocol = verify(root)
    summaries = ARMS.to_h do |arm|
      result = JSON.parse(File.read(File.join(root, "#{arm}-1337/summary.json")))
      raise "Pilot budget incomplete" unless result.fetch("step") == STEPS
      [arm, result]
    end
    selected = summaries.filter_map { |arm, result| [arm, result.fetch("selected")] if result.fetch("selected") }
    winner = selected.max_by { |_, result| result.fetch("key") }
    baselines = JSON.parse(File.read(File.join(root, "baselines.json")))
    reference = baselines.fetch("models").fetch(baselines.fetch("chosen")).fetch("validation").fetch("factual_macro_accuracy")
    run_confirmation = winner && winner.last.fetch("metrics").fetch("factual_macro_accuracy") >= reference + protocol.fetch("confirmation").fetch("minimum_validation_gain")
    decision = { "winner" => winner&.first, "confirmation_required" => !!run_confirmation, "reference_validation_macro" => reference,
      "criterion" => protocol.fetch("confirmation"), "scope" => "Validation-only decision before final evaluation" }
    path = File.join(root, "confirmation.json")
    raise "Confirmation decision already frozen" if File.exist?(path)
    write(path, decision)
    train(root, winner.first, protocol.fetch("confirmation").fetch("seed")) if run_confirmation
  end

  def calibration_logits(root, loaded, device)
    raw = raw_rows(File.join(root, "data/calibration-clean.jsonl"))
    rows = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    evaluator = SemanticCoverageEvaluation.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), device: device, batch_size: 4)
    [rows, evaluator.collect(rows)]
  end

  def final_candidates(root)
    confirmation = JSON.parse(File.read(File.join(root, "confirmation.json")))
    candidates = INITIALIZERS.dup
    ARMS.each do |arm|
      result = JSON.parse(File.read(File.join(root, "#{arm}-1337/summary.json")))
      raise "Pilot incomplete" unless result.fetch("step") == STEPS
      candidates["#{arm}-1337"] = result.fetch("selected")&.fetch("checkpoint") || File.join(root, "#{arm}-1337/choice")
    end
    if confirmation.fetch("confirmation_required")
      name = "#{confirmation.fetch('winner')}-2027"
      result = JSON.parse(File.read(File.join(root, name, "summary.json")))
      raise "Confirmation incomplete" unless result.fetch("step") == STEPS
      candidates[name] = result.fetch("selected")&.fetch("checkpoint") || File.join(root, name, "choice")
    end
    candidates
  end

  def evaluate(root)
    verify(root)
    candidates = final_candidates(root)
    marker = File.join(root, "acceptance-opened.json")
    raise "Final panel already opened; no retuning on this test" if File.exist?(marker)
    verify_panels(root)
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    policies = candidates.to_h do |name, path|
      puts "Calibrating #{name} on the clean calibration panel"
      loaded = EasyAI::Decision::Checkpoint.load(path)
      rows, logits = calibration_logits(root, loaded, device)
      calibrator = EasyAI::Decision::Calibrator.new.fit(logits, rows.map(&:target_index))
      value = { "checkpoint" => loaded.fetch(:path), "weights_sha256" => loaded.fetch(:weights_fingerprint), "temperature" => calibrator.temperature }
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      [name, value]
    end
    write(File.join(root, "calibration.json"), policies)
    write(marker, { "opened_at" => Time.now.utc.iso8601, "policies_sha256" => Digest::SHA256.file(File.join(root, "calibration.json")).hexdigest,
      "provenance_sha256" => Digest::SHA256.file(File.join(root, "provenance.json")).hexdigest,
      "confirmation_sha256" => Digest::SHA256.file(File.join(root, "confirmation.json")).hexdigest,
      "scope" => "All objectives, checkpoints, confirmation and temperatures frozen before evaluating this panel" })
    raw = raw_rows(File.join(root, "data/acceptance.jsonl"))
    results = policies.to_h do |name, policy|
      loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
      raise "Calibrated weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
      calibrator = EasyAI::Decision::Calibrator.new(temperature: policy.fetch("temperature"))
      metrics = measure(loaded.fetch(:model), loaded.fetch(:tokenizer), raw, device, calibrator: calibrator,
        predictions_path: File.join(root, "predictions-#{name}.jsonl"), include_raw: true)
      write(File.join(root, "evaluation-#{name}.json"), metrics)
      puts "Final #{name}: factual=#{metrics.fetch('factual_macro_accuracy').round(4)} pairs=#{metrics.fetch('pair_all_correct').round(4)}"
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      [name, metrics]
    end
    write(File.join(root, "evaluation.json"), results)
  end
end
