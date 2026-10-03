module DecisionJudgment
  module_function

  def baseline(root)
    verify(root)
    raise "Baseline exists" if File.exist?(File.join(root, "baseline.json"))
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    value = %w[binary joint].to_h { |profile| [profile, measure(loaded.fetch(:model), loaded.fetch(:tokenizer), panel(root, "validation", profile), DecisionRobust.device)] }
    write(File.join(root, "baseline.json"), value)
    write(File.join(root, "baseline-snapshot.json"), { "protocol_sha256" => sha(File.join(root, "protocol.json")), "baseline_sha256" => sha(File.join(root, "baseline.json")) })
    puts "Parent canonical development: #{value.transform_values { |cell| cell.fetch('worst_language_balanced') }}"
  end

  def initialize_trainer(root, arm, seed, fitting: false)
    protocol = verify(root)
    raise ArgumentError, "Unknown arm" unless ARMS.include?(arm)
    directory = File.join(root, fitting ? "fit" : "#{arm}-#{seed}")
    cfg = EasyAI::Decision::Config.new(protocol.fetch("config")).with(training: { seed: seed })
    cfg = cfg.with(model: { dropout: 0.0 }, training: { learning_rate: 0.0003, warmup_steps: 0 }) if fitting
    path = File.join(root, fitting ? "data/fit.jsonl" : "data/#{arm}-train.jsonl")
    dataset = EasyAI::Decision::Data::Dataset.new(path)
    klass = fitting ? EasyAI::Decision::CandidateTrainer : EasyAI::Decision::JudgmentTrainer
    output = File.join(directory, "choice")
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    if File.exist?(File.join(output, "latest.json"))
      trainer = klass.resume(output, dataset: dataset, output: output)
      raise "Resume config mismatch" unless trainer.model.config.to_h == cfg.to_h
      return trainer
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    model = EasyAI::Decision::ChoiceModel.new(cfg)
    model.load_state_dict(loaded.fetch(:model).state_dict)
    klass.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output)
  end

  def fit(root)
    protocol = verify(root)
    raise "Fitting result exists" if File.exist?(File.join(root, "fit/result.json"))
    trainer = initialize_trainer(root, "candidate", 1337, fitting: true)
    rows = raw(File.join(root, "data/fit.jsonl"))
    measurements = []
    stable = 0
    (100..protocol.fetch("fit").fetch("maximum_steps")).step(100) do |budget|
      trainer.train(steps: budget)
      value = measure(trainer.model, trainer.tokenizer, rows, trainer.device).merge("step" => budget)
      measurements << value
      passed = value.fetch("by_language").values.all? { |cell| cell.fetch("accuracy") >= protocol.dig("fit", "accuracy") && cell.fetch("binding_all_correct") >= protocol.dig("fit", "mixed_groups") }
      stable = passed ? stable + 1 : 0
      write(File.join(root, "fit/measurements.json"), measurements)
      puts "Fit #{budget}: accuracy=#{value.fetch('accuracy').round(4)} mixed=#{value.fetch('by_language').transform_values { |cell| cell.fetch('binding_all_correct') }}"
      break if stable >= 2
    end
    _, final_logits = DecisionRobust.collect(trainer.model, trainer.tokenizer, rows, trainer.device)
    DecisionRobust.write_rows(File.join(root, "fit/predictions.jsonl"), rows.each_with_index.map { |row, index| row.merge("logits" => final_logits[index]) })
    parent = EasyAI::Decision::Checkpoint.load(PARENT)
    change = (trainer.model.encoder.embedding.weight.detach.cpu - parent.fetch(:model).encoder.embedding.weight.detach.cpu).abs.max.item
    write(File.join(root, "fit/result.json"), { "passed" => stable >= 2, "measurements" => measurements, "embedding_max_change" => change,
      "last_checkpoint" => trainer.last_checkpoint, "device" => trainer.device, "scope" => "Two complete training-only frames; fitting weights never initialize pilots" })
  end

  def mastery?(training, development, rules)
    %w[en-US zh-CN].all? do |language|
      train = training.fetch("by_language").fetch(language)
      dev = development.fetch("by_language").fetch(language)
      train.fetch("balanced") >= rules.fetch("known_balanced") && train.fetch("binding_groups").positive? &&
        train.fetch("binding_all_correct") >= rules.fetch("mixed_groups") && dev.fetch("balanced") >= rules.fetch("dev_balanced")
    end
  end

  def train(root, arm, seed)
    protocol = verify(root)
    raise "Fit failed" unless JSON.parse(File.read(File.join(root, "fit/result.json"))).fetch("passed")
    directory = File.join(root, "#{arm}-#{seed}")
    raise "Completed pilot exists" if File.exist?(File.join(directory, "summary.json"))
    trainer = initialize_trainer(root, arm, seed)
    probe = raw(File.join(root, "data/#{arm}-probe.jsonl"))
    history_path = File.join(directory, "validation.json")
    history = File.exist?(history_path) ? JSON.parse(File.read(history_path)) : []
    # Roll back uncommitted observations after interruption to the checkpoint boundary.
    history.select! { |row| row.fetch("step") <= trainer.state.fetch("step") }
    observed = trainer.state.fetch("best_observed", {})
    selected = trainer.state.fetch("selected", {})
    trainer.train do |state, loss|
      puts "#{arm}/#{seed} step=#{state['step']} loss=#{loss.round(4)} stage=#{state['judgment_stage']} device=#{trainer.device}" if (state.fetch("step") % 20).zero?
      next unless (state.fetch("step") % 100).zero?
      before = trainer.model.named_parameters.transform_values(&:object_id)
      parameter = trainer.model.named_parameters.fetch("encoder.embedding.weight").detach.cpu.clone
      stage = state.fetch("judgment_stage")
      values = { "binary" => measure(trainer.model, trainer.tokenizer, panel(root, "validation", "binary"), trainer.device) }
      values["joint"] = measure(trainer.model, trainer.tokenizer, panel(root, "validation", "joint"), trainer.device) if stage == "joint"
      probe_value = measure(trainer.model, trainer.tokenizer, probe, trainer.device)
      raise "Validation replaced live parameters" unless before == trainer.model.named_parameters.transform_values(&:object_id)
      raise "Validation changed model weights" unless (parameter - trainer.model.named_parameters.fetch("encoder.embedding.weight").detach.cpu).abs.max.item.zero?
      values.each do |profile, value|
        key = [value.fetch("worst_language_balanced"), -value.fetch("nll")]
        previous = observed[profile]
        allowed = eligible?(value, profile, protocol.fetch("publication"))
        best = selected[profile]
        if previous.nil? || (key <=> previous.fetch("key")) == 1 || (allowed && (best.nil? || (key <=> best.fetch("key")) == 1))
          checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "observed-#{profile}"), model: trainer.model, tokenizer: trainer.tokenizer, training_state: state)
          choice = { "checkpoint" => checkpoint, "key" => key, "step" => state.fetch("step"), "metrics" => value }
          observed[profile] = choice if previous.nil? || (key <=> previous.fetch("key")) == 1
          selected[profile] = choice if allowed && (best.nil? || (key <=> best.fetch("key")) == 1)
        end
      end
      mastered = stage == "known" && mastery?(probe_value, values.fetch("binary"), protocol.fetch("mastery"))
      state["mastery_checks"] = mastered ? state.fetch("mastery_checks") + 1 : 0 if stage == "known"
      entry = { "step" => state.fetch("step"), "stage" => stage, "profiles" => values, "training_probe" => probe_value,
        "mastery_checks" => state.fetch("mastery_checks"), "device" => trainer.device,
        "gpu_process_mib" => trainer.device == "cuda" ? EasyAI::Runtime::DevicePolicy.new(requested: "cuda").process_memory_mib : nil }
      history << entry
      state["best_observed"], state["selected"] = observed, selected
      write(history_path, history)
      if stage == "known"
        if state.fetch("mastery_checks") >= protocol.dig("mastery", "stable_checks")
          state["pre_unknown_checkpoint"] = trainer.save_checkpoint
          trainer.advance!(mastered: true)
        elsif state.fetch("step") >= protocol.fetch("known_budget")
          state["stop_reason"] = "Known-stage mastery failed within the frozen 1000-update budget"
        end
      end
      puts "#{arm}/#{seed} dev=#{values.transform_values { |value| value.fetch('worst_language_balanced').round(4) }} training=#{probe_value.fetch('by_language').transform_values { |cell| cell.slice('balanced', 'binding_all_correct') }} mastery=#{state['mastery_checks']}"
    end
    write(File.join(directory, "summary.json"), { "step" => trainer.state.fetch("step"), "stage" => trainer.state.fetch("judgment_stage"),
      "stop_reason" => trainer.state["stop_reason"], "best_observed" => observed, "selected" => selected,
      "last_checkpoint" => trainer.last_checkpoint, "device" => trainer.device, "examples_seen" => trainer.state.fetch("examples_seen"), "coverage" => trainer.state.fetch("coverage") })
  end

  def confirm(root)
    protocol = verify(root)
    raise "Confirmation decision exists" if File.exist?(File.join(root, "confirmation.json"))
    baseline = JSON.parse(File.read(File.join(root, "baseline.json")))
    candidates = ARMS.flat_map do |arm|
      path = File.join(root, "#{arm}-1337/summary.json")
      next [] unless File.exist?(path)
      summary = JSON.parse(File.read(path))
      summary.fetch("selected").filter_map do |profile, choice|
        gain = %w[en-US zh-CN].all? { |language| choice.dig("metrics", "by_language", language, "balanced") >= baseline.dig(profile, "by_language", language, "balanced") + protocol.dig("publication", "parent_gain") }
        [arm, profile, choice] if gain
      end
    end
    winner = candidates.max_by { |_, _, choice| choice.fetch("key") }
    decision = { "confirmation_required" => !!winner, "winner" => winner&.first, "profile" => winner&.[](1), "choice" => winner&.last,
      "scope" => "Development-only profile selection; no acceptance read", "seed" => 2027 }
    write(File.join(root, "confirmation.json"), decision)
    train(root, winner.first, 2027) if winner
    puts JSON.pretty_generate(decision.slice("confirmation_required", "winner", "profile"))
  end
end
