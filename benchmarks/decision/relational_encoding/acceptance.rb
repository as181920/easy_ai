module DecisionRelational
  module_function

  def acceptance_views(raw)
    result = DecisionJudgment.views(raw)
    result["reversed_record"] = raw.map do |row|
      chinese = row.fetch("language") == "zh-CN"
      clauses = chinese ? row.fetch("state").split("。").reject(&:empty?) : row.fetch("state").delete_suffix(".").split(". ")
      row.merge("state" => clauses.reverse.join(chinese ? "。" : ". ") + (chinese ? "。" : "."))
    end
    result
  end

  def accepted_case(directory, policy, profile)
    loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
    raise "Accepted checkpoint weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
    predictions = []
    raw = DecisionJudgment.panel(directory, "test", profile)
    logits_by_view = {}
    values = acceptance_views(raw).to_h do |view, records|
      _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), records, DecisionRobust.device)
      logits_by_view[view] = logits
      records.each_with_index { |row, index| predictions << row.merge("view" => view, "logits" => logits[index]) }
      [view, { "raw" => DecisionJudgment.metrics(records, logits),
        "calibrated" => DecisionJudgment.metrics(records, logits, temperature: policy.fetch("temperature")),
        "device" => loaded.fetch(:model).parameters.first.device.type.to_s }]
    end
    natural = rows(File.join(directory, "data/natural-test.jsonl")).select { |row| row.fetch("profile") == profile }
    _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), natural, DecisionRobust.device)
    natural.each_with_index { |row, index| predictions << row.merge("view" => "natural", "logits" => logits[index]) }
    values["natural"] = { "raw" => DecisionJudgment.metrics(natural, logits),
      "calibrated" => DecisionJudgment.metrics(natural, logits, temperature: policy.fetch("temperature")) }
    loaded.fetch(:model).to("cpu")
    loaded = nil
    GC.start
    { "metrics" => values, "predictions" => predictions, "canonical_logits" => logits_by_view.fetch("canonical"), "natural_logits" => logits }
  end

  def accept(root, variant: "standard")
    protocol = pilot_protocol(root, variant: variant)
    directory = pilot_directory(root, variant)
    raise "Acceptance already evaluated" if File.exist?(File.join(directory, "acceptance.json"))
    calibrate(root, variant: variant) unless File.exist?(File.join(directory, "calibration.json"))
    policy = frozen_acceptance(root, variant)
    profile = policy.fetch("profile")
    cases = policy.fetch("policies").to_h do |name, choice|
      value = accepted_case(directory, choice, profile)
      DecisionRobust.write_rows(File.join(directory, "acceptance-#{name}.jsonl"), value.fetch("predictions"))
      puts "Acceptance #{name}: #{value.dig('metrics', 'canonical', 'raw', 'by_language').transform_values { |cell| cell.slice('balanced', 'actor_binding_all_correct', 'event_binding_all_correct') }}"
      $stdout.flush
      [name, value]
    end
    parent = cases.fetch("parent")
    baseline = { "controlled" => parent.dig("metrics", "canonical", "raw"), "natural" => parent.dig("metrics", "natural", "raw") }
    canonical = DecisionJudgment.panel(directory, "test", profile)
    natural = rows(File.join(directory, "data/natural-test.jsonl")).select { |row| row.fetch("profile") == profile }
    checks = %w[candidate confirmation].to_h do |name|
      value = cases.fetch(name)
      assessed = { "controlled" => value.dig("metrics", "canonical", "raw"), "natural" => value.dig("metrics", "natural", "raw") }
      value.fetch("metrics")["canonical_gain_uncertainty"] = paired_uncertainty(canonical, parent.fetch("canonical_logits"), value.fetch("canonical_logits"))
      value.fetch("metrics")["natural_gain_uncertainty"] = paired_uncertainty(natural, parent.fetch("natural_logits"), value.fetch("natural_logits"))
      [name, pilot_eligible?(assessed, baseline, profile, protocol.fetch("publication"))]
    end
    output = { "passed" => checks.values.all?, "checks" => checks, "profile" => profile,
      "cases" => cases.transform_values { |value| value.fetch("metrics") },
      "calibration_sha256" => sha(File.join(directory, "calibration.json")),
      "scope" => "One reserved acceptance opening with fixed primary/confirmation checkpoints. Canonical and natural floors gate publication; held wording/options/order are supplementary. No post-acceptance checkpoint or threshold selection." }
    write(File.join(directory, "acceptance.json"), output)
    output
  end

  def package_preview(root, variant: "standard")
    directory = pilot_directory(root, variant)
    policy = frozen_acceptance(root, variant)
    result = JSON.parse(File.read(File.join(directory, "acceptance.json")))
    raise "Acceptance failed; no preview" unless result.fetch("passed") && result.fetch("checks").values.all?
    raise "Acceptance calibration changed" unless result.fetch("calibration_sha256") == sha(File.join(directory, "calibration.json"))
    destination = File.join(directory, "factual-preview")
    raise "Preview already exists" if File.exist?(destination)
    choice = policy.fetch("policies").fetch("candidate")
    loaded = EasyAI::Decision::Checkpoint.load(choice.fetch("checkpoint"))
    raise "Preview weights changed" unless loaded.fetch(:weights_fingerprint) == choice.fetch("weights_sha256")
    checkpoint = EasyAI::Decision::Checkpoint.save(destination, model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer),
      training_state: { "step" => loaded.fetch(:metadata).fetch("training").fetch("step"), "source_weights_sha256" => loaded.fetch(:weights_fingerprint) },
      calibration: { "temperature" => choice.fetch("temperature"), "profile" => policy.fetch("profile"), "scope" => policy.fetch("capability") })
    write(File.join(destination, "preview.json"), { "checkpoint" => checkpoint, "capability" => policy.fetch("capability"),
      "acceptance_sha256" => sha(File.join(directory, "acceptance.json")), "profile" => policy.fetch("profile"),
      "scope" => "Local factual inference preview only. The delivered v0.1 routing preview is not replaced." })
    puts "Packaged #{destination}"
    destination
  end
end
