module DecisionJudgment
  module_function

  def metrics(rows, logits, temperature: 1.0)
    raise ArgumentError, "Prediction count mismatch" unless rows.size == logits.size && !rows.empty?
    examples = rows.map { |row| EasyAI::Decision::Data::Example.new(row) }
    calibrator = EasyAI::Decision::Calibrator.new(temperature: temperature)
    result = EasyAI::Decision::Evaluator.metrics(logits, examples.map(&:target_index), calibrator)
    predictions = examples.each_with_index.map { |row, index| row.options.fetch(logits[index].each_index.max_by { |option| logits[index][option] }).fetch("id") }
    result["by_language"] = rows.each_index.group_by { |index| rows[index].fetch("language") }.transform_values do |indexes|
      subset = EasyAI::Decision::Evaluator.metrics(indexes.map { |i| logits[i] }, indexes.map { |i| examples[i].target_index }, calibrator)
      classes = rows[indexes.first].fetch("options").map { |option| option.fetch("id") }
      recall = classes.to_h do |target|
        selected = indexes.select { |i| rows[i].fetch("target") == target }
        [target, selected.empty? ? nil : selected.count { |i| predictions[i] == target }.fdiv(selected.size)]
      end
      subset["recall"] = recall
      subset["balanced"] = recall.values.any?(&:nil?) ? nil : recall.values.sum.fdiv(recall.size)
      subset["confusion"] = indexes.group_by { |i| rows[i].fetch("target") }.transform_values { |items| items.map { |i| predictions[i] }.tally }
      %w[binding fact_flip question_flip].each do |axis|
        groups = indexes.select { |i| rows[i].dig("contrast_groups", axis) }.group_by { |i| rows[i].dig("contrast_groups", axis) }
        raise ArgumentError, "Incomplete #{axis} evaluation group" unless groups.values.all? { |items| items.size == 2 }
        if axis == "binding"
          groups.select! do |_, items|
            row = rows[items.first]
            world = row.fetch("world", {})
            world["version"] == 4 ? world.fetch("mixed") : world.fetch("facts", {}).values.uniq.size == 2
          end
        end
        subset["#{axis}_groups"] = groups.size
        subset["#{axis}_all_correct"] = groups.empty? ? nil : groups.values.count { |items| items.all? { |i| predictions[i] == rows[i].fetch("target") } }.fdiv(groups.size)
        next unless axis == "binding"
        %w[actor event].each do |kind|
          selected = groups.values.select do |items|
            world = rows[items.first].fetch("world")
            actual = world.fetch("version") == 4 && world.fetch("facts").map { |fact| fact.fetch("actor") }.uniq.size == 1 ? "event" : "actor"
            actual == kind
          end
          subset["#{kind}_binding_groups"] = selected.size
          subset["#{kind}_binding_all_correct"] = selected.empty? ? nil : selected.count { |items| items.all? { |i| predictions[i] == rows[i].fetch("target") } }.fdiv(selected.size)
        end
      end
      subset
    end
    values = result.fetch("by_language").values.map { |row| row.fetch("balanced") }
    result["worst_language_balanced"] = values.any?(&:nil?) ? nil : values.min
    result
  end

  def measure(model, tokenizer, rows, device)
    _, logits = DecisionRobust.collect(model, tokenizer, rows, device)
    metrics(rows, logits)
  end

  def eligible?(value, profile, gates)
    value.fetch("by_language").keys.sort == %w[en-US zh-CN] && value.fetch("by_language").values.all? do |cell|
      next false unless cell.fetch("binding_groups").positive? && cell.fetch("binding_all_correct") >= gates.fetch("mixed_groups")
      next false unless %w[actor event].all? { |kind| cell.fetch("#{kind}_binding_groups").positive? && cell.fetch("#{kind}_binding_all_correct") >= gates.fetch("mixed_groups") }
      if profile == "binary"
        cell.fetch("balanced") && cell.fetch("balanced") >= gates.fetch("binary_balanced")
      else
        %w[yes no].all? { |label| cell.fetch("recall")[label] && cell.fetch("recall")[label] >= gates.fetch("joint_known_recall") } &&
          cell.fetch("recall")["unknown"] && cell.fetch("recall")["unknown"] >= gates.fetch("joint_unknown_recall")
      end
    end
  end

  def panel(root, name, profile)
    raw(File.join(root, "data/#{name}.jsonl")).select { |row| row.dig("world", "profile") == profile }
  end

  def views(rows)
    { "canonical" => rows,
      "held_expression" => rows.map do |row|
        language = row.fetch("language")
        row.merge("state" => language == "zh-CN" ? "日志记载如下：#{row.fetch('state')}" : "The log records the following: #{row.fetch('state')}",
          "question" => language == "zh-CN" ? row.fetch("question").sub("根据记录判断陈述：", "日志能否证实下面的说法：") : row.fetch("question").sub("Judge this claim against the record: ", "Does the log establish the following claim? "))
      end,
      "held_options" => rows.map { |row| row.merge("options" => EasyAI::Decision::Data::JudgmentCorpus.options(row.fetch("language"), binary: row.fetch("options").size == 2, held: true)) } }
  end

  def freeze_policy(root)
    protocol = verify(root)
    raise "Policy already frozen" if File.exist?(File.join(root, "calibration.json"))
    decision = JSON.parse(File.read(File.join(root, "confirmation.json")))
    fit = JSON.parse(File.read(File.join(root, "fit/result.json")))
    raise "Fitting failed; acceptance must remain closed" unless fit.fetch("passed")
    if fit.fetch("passed")
      ARMS.each do |arm|
        path = File.join(root, "#{arm}-1337/summary.json")
        raise "Pilot incomplete" unless File.exist?(path)
        summary = JSON.parse(File.read(path))
        terminal = summary.fetch("step") == protocol.fetch("steps") ||
          (summary.fetch("step") == protocol.fetch("known_budget") && summary.fetch("stage") == "known" && summary["stop_reason"])
        raise "Pilot incomplete" unless terminal
      end
    end
    choices = { "parent" => PARENT, "v03" => "runs/decision/factual-v03/pilot/candidate-1337/choice" }
    ARMS.each do |arm|
      path = File.join(root, "#{arm}-1337/summary.json")
      next unless File.exist?(path)
      summary = JSON.parse(File.read(path))
      %w[binary joint].each { |profile| choices["#{arm}-#{profile}"] = summary.fetch("best_observed").dig(profile, "checkpoint") }
    end
    if decision.fetch("confirmation_required")
      choices["confirmation"] = JSON.parse(File.read(File.join(root, "#{decision.fetch('winner')}-2027/summary.json"))).fetch("selected").dig(decision.fetch("profile"), "checkpoint")
    end
    policies = choices.compact.to_h do |name, path|
      loaded = EasyAI::Decision::Checkpoint.load(path)
      profiles = %w[binary joint].to_h do |profile|
        rows = panel(root, "calibration", profile)
        examples, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, DecisionRobust.device)
        calibrator = EasyAI::Decision::Calibrator.new.fit(logits, examples.map(&:target_index))
        [profile, { "temperature" => calibrator.temperature, "raw" => metrics(rows, logits), "calibrated" => metrics(rows, logits, temperature: calibrator.temperature) }]
      end
      loaded.fetch(:model).to("cpu")
      value = { "checkpoint" => loaded.fetch(:path), "weights_sha256" => loaded.fetch(:weights_fingerprint), "profiles" => profiles }
      loaded = nil
      GC.start
      [name, value]
    end
    write(File.join(root, "calibration.json"), policies)
    write(File.join(root, "acceptance-opened.json"), { "at" => Time.now.utc.iso8601, "protocol_sha256" => sha(File.join(root, "protocol.json")),
      "policies_sha256" => sha(File.join(root, "calibration.json")), "confirmation_sha256" => sha(File.join(root, "confirmation.json")), "data_sha256" => protocol.fetch("files_sha256") })
    policies
  end

  def verify_opened(root)
    protocol = verify(root)
    marker = JSON.parse(File.read(File.join(root, "acceptance-opened.json")))
    raise "Opened policy changed" unless marker.fetch("protocol_sha256") == sha(File.join(root, "protocol.json")) &&
      marker.fetch("policies_sha256") == sha(File.join(root, "calibration.json")) && marker.fetch("confirmation_sha256") == sha(File.join(root, "confirmation.json"))
    protocol
  end

  def evaluate(root)
    policies = File.exist?(File.join(root, "calibration.json")) ? JSON.parse(File.read(File.join(root, "calibration.json"))) : freeze_policy(root)
    verify_opened(root)
    raise "Acceptance already evaluated" if File.exist?(File.join(root, "evaluation.json"))
    results = policies.to_h do |name, policy|
      loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
      raise "Selected weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
      predictions = []
      profile_results = %w[binary joint].to_h do |profile|
        values = views(panel(root, "test", profile)).to_h do |view, rows|
          _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, DecisionRobust.device)
          temperature = policy.fetch("profiles").fetch(profile).fetch("temperature")
          rows.each_with_index { |row, index| predictions << row.merge("view" => view, "logits" => logits[index]) }
          [view, { "raw" => metrics(rows, logits), "calibrated" => metrics(rows, logits, temperature: temperature) }]
        end
        [profile, values]
      end
      DecisionRobust.write_rows(File.join(root, "predictions-#{name}.jsonl"), predictions)
      regression = raw(File.join(root, "data/regression.jsonl"))
      regression_value = SemanticCoverageEvaluation.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), device: DecisionRobust.device, batch_size: 4)
        .evaluate(regression.map { |row| EasyAI::Decision::Data::Example.new(row) })
      profile_results["public_and_routing_regression"] = regression_value
      profile_results["natural"] = %w[binary joint].to_h do |profile|
        natural = raw(File.join(root, "data/natural-test.jsonl")).select { |row| row.fetch("profile") == profile }
        _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), natural, DecisionRobust.device)
        natural.each_with_index { |row, index| predictions << row.merge("view" => "natural", "logits" => logits[index]) }
        [profile, metrics(natural, logits)]
      end
      DecisionRobust.write_rows(File.join(root, "predictions-#{name}.jsonl"), predictions)
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      puts "Acceptance #{name}: #{profile_results.slice('binary', 'joint').transform_values { |views| views.fetch('canonical').fetch('raw').fetch('worst_language_balanced') }}"
      [name, profile_results]
    end
    write(File.join(root, "evaluation.json"), results)
  end
end
