module DecisionRelational
  # Same batches and numerical evaluation; bound deferred tensor collection.
  EVALUATION_GC_INTERVAL = 8
  module_function

  def pilot_measure(model, tokenizer, directory, profile, device)
    controlled = DecisionJudgment.panel(directory, "validation", profile)
    natural = rows(File.join(directory, "data/natural-development.jsonl")).select { |row| row.fetch("profile") == profile }
    { "controlled" => DecisionJudgment.measure(model, tokenizer, controlled, device, gc_interval: EVALUATION_GC_INTERVAL),
      "natural" => DecisionJudgment.measure(model, tokenizer, natural, device, gc_interval: EVALUATION_GC_INTERVAL) }
  end

  def pilot_baseline(root, variant: "standard")
    pilot_protocol(root, variant: variant)
    directory = pilot_directory(root, variant)
    path = File.join(directory, "baseline.json")
    if File.exist?(path)
      expected = JSON.parse(File.read(File.join(directory, "baseline-sha256.json"))).fetch("sha256")
      raise "Parent baseline changed" unless sha(path) == expected
      return JSON.parse(File.read(path))
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    device = DecisionRobust.device
    values = %w[binary joint].to_h { |profile| [profile, pilot_measure(loaded.fetch(:model), loaded.fetch(:tokenizer), directory, profile, device)] }
    write(path, values)
    write(File.join(directory, "baseline-sha256.json"), { "sha256" => sha(path) })
    values
  end

  def known_mastery?(training, development, rules)
    %w[en-US zh-CN].all? do |language|
      train = training.fetch("by_language").fetch(language)
      dev = development.fetch("by_language").fetch(language)
      train.fetch("balanced") && train.fetch("balanced") >= rules.fetch("known_balanced") &&
        dev.fetch("balanced") && dev.fetch("balanced") >= rules.fetch("dev_balanced") && %w[actor event].all? do |kind|
          train.fetch("#{kind}_binding_groups").positive? && train.fetch("#{kind}_binding_all_correct") >= rules.fetch("mixed_groups")
        end
    end
  end

  def pilot_eligible?(value, baseline, profile, rules)
    return false unless DecisionJudgment.eligible?(value.fetch("controlled"), profile, rules)
    %w[en-US zh-CN].all? do |language|
      controlled = value.dig("controlled", "by_language", language)
      reference = baseline.dig("controlled", "by_language", language)
      natural = value.dig("natural", "by_language", language)
      parent = baseline.dig("natural", "by_language", language)
      controlled.fetch("balanced") >= reference.fetch("balanced") + rules.fetch("parent_gain") &&
        natural.fetch("accuracy") >= rules.fetch("natural_minimum_accuracy") &&
        natural.fetch("accuracy") >= parent.fetch("accuracy") - rules.fetch("natural_tolerance")
    end
  end

  def pilot_trainer(root, seed, protocol, variant: "standard")
    directory = File.join(pilot_directory(root, variant), "seed-#{seed}")
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    output = File.join(directory, "choice")
    config = EasyAI::Decision::Config.new(protocol.fetch("config")).with(training: { seed: seed })
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(pilot_directory(root, variant), "data/candidate-train.jsonl"))
    klass = focused_variant?(variant) ? FocusedTrainer : EasyAI::Decision::JudgmentTrainer
    if File.exist?(File.join(output, "latest.json"))
      trainer = klass.resume(output, dataset: dataset, output: output)
      raise "Pilot resume config changed" unless trainer.model.config.to_h == config.to_h
      return trainer
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    raise "Unexpected scratch pilot" unless protocol.fetch("initialization") == "parent"
    model, manifest = EasyAI::Decision::WeightTransfer.apply(parent: loaded, config: config, tokenizer: loaded.fetch(:tokenizer), seed: seed)
    write(File.join(directory, "transfer.json"), manifest)
    klass.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output)
  end

  def pilot(root, seed, variant: "standard")
    raise ArgumentError, "Unsupported pilot seed" unless [1337, 2027].include?(seed)
    protocol = pilot_protocol(root, variant: variant)
    known_budget = protocol.fetch("known_budget", 1000)
    if seed == 2027
      selection = JSON.parse(File.read(File.join(pilot_directory(root, variant), "selection.json")))
      raise "No eligible pilot confirmation" unless selection.fetch("profile")
    end
    directory = File.join(pilot_directory(root, variant), "seed-#{seed}")
    raise "Pilot completed" if File.exist?(File.join(directory, "summary.json"))
    baseline = pilot_baseline(root, variant: variant)
    trainer = pilot_trainer(root, seed, protocol, variant: variant)
    probe = rows(File.join(pilot_directory(root, variant), "data/candidate-probe.jsonl"))
    history_path = File.join(directory, "development.json")
    history = File.exist?(history_path) ? JSON.parse(File.read(history_path)).select { |cell| cell.fetch("step") <= trainer.state.fetch("step") } : []
    selected = trainer.state.fetch("selected", {})
    blocked = trainer.state.fetch("judgment_stage") == "known" && trainer.state.fetch("step") >= known_budget && !trainer.state.fetch("known_mastery_passed", false)
    trainer.state["stop_reason"] = "Known-stage mastery failed by the #{known_budget}-update ceiling" if blocked
    enforce_update_budget!(root, [protocol.dig("config", "training", "steps") - trainer.state.fetch("step"), 0].max)
    unless blocked
      trainer.train do |state, loss|
        if (state.fetch("step") % 20).zero?
          puts "pilot/#{seed} step=#{state.fetch('step')} loss=#{loss.round(5)} stage=#{state.fetch('judgment_stage')} device=#{trainer.device}"
          $stdout.flush
        end
        next unless (state.fetch("step") % 100).zero?
        stage = state.fetch("judgment_stage")
        profiles = stage == "known" ? ["binary"] : %w[binary joint]
        identities = trainer.model.named_parameters.transform_values(&:object_id)
        values = profiles.to_h { |profile| [profile, pilot_measure(trainer.model, trainer.tokenizer, pilot_directory(root, variant), profile, trainer.device)] }
        training = DecisionJudgment.measure(trainer.model, trainer.tokenizer, probe, trainer.device, gc_interval: EVALUATION_GC_INTERVAL)
        raise "Pilot evaluation replaced parameters" unless trainer.model.named_parameters.transform_values(&:object_id) == identities
        values.each do |profile, value|
          key = [value.dig("controlled", "worst_language_balanced"), -value.dig("controlled", "nll")]
          selectable = true # Binary and three-state contracts are independently assessed.
          eligible = selectable && pilot_eligible?(value, baseline.fetch(profile), profile, protocol.fetch("publication"))
          next unless eligible && (!selected[profile] || (key <=> selected.fetch(profile).fetch("key")) == 1)
          checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "selected-#{profile}"), model: trainer.model, tokenizer: trainer.tokenizer, training_state: state)
          selected[profile] = { "checkpoint" => checkpoint, "step" => state.fetch("step"), "key" => key, "metrics" => value }
        end
        mastered = stage == "known" && known_mastery?(training, values.fetch("binary").fetch("controlled"), protocol.fetch("mastery"))
        state["mastery_checks"] = mastered ? state.fetch("mastery_checks") + 1 : 0 if stage == "known"
        state["selected"] = selected
        history << { "step" => state.fetch("step"), "stage" => stage, "profiles" => values, "training_probe" => training,
          "mastery_checks" => state.fetch("mastery_checks"), "device" => trainer.device }
        write(history_path, history)
        if stage == "known"
          if state.fetch("mastery_checks") >= 2
            state["known_mastery_passed"] = true
            state["pre_unknown_checkpoint"] = trainer.save_checkpoint
            trainer.advance!(mastered: true) if state.fetch("step") < protocol.fetch("config").fetch("training").fetch("steps")
          elsif state.fetch("step") >= known_budget
            state["stop_reason"] = "Known-stage mastery failed by the #{known_budget}-update ceiling"
          end
        end
        puts "pilot/#{seed} development=#{values.transform_values { |value| value.dig('controlled', 'worst_language_balanced') }} training=#{training.fetch('by_language').transform_values { |cell| cell.slice('balanced', 'actor_binding_all_correct', 'event_binding_all_correct') }} eligible=#{selected.keys}"
        $stdout.flush
      end
    end
    write(File.join(directory, "summary.json"), { "step" => trainer.state.fetch("step"), "stage" => trainer.state.fetch("judgment_stage"),
      "known_mastery_passed" => trainer.state.fetch("known_mastery_passed", false), "selected" => trainer.state["known_mastery_passed"] ? selected : {},
      "stop_reason" => trainer.state["stop_reason"], "last_checkpoint" => trainer.last_checkpoint, "device" => trainer.device,
      "known_updates" => trainer.state.fetch("joint_started_at", trainer.state.fetch("step")),
      "unknown_updates" => trainer.state["joint_started_at"] ? trainer.state.fetch("step") - trainer.state.fetch("joint_started_at") : 0,
      "examples_seen" => trainer.state.fetch("examples_seen"), "coverage" => trainer.state.fetch("coverage") })
  end

  def select_pilot(root, variant: "standard")
    pilot_protocol(root, variant: variant)
    directory = pilot_directory(root, variant)
    path = File.join(directory, "selection.json")
    raise "Pilot selection already frozen" if File.exist?(path)
    summary = JSON.parse(File.read(File.join(directory, "seed-1337/summary.json")))
    choice = summary.fetch("selected").max_by { |profile, cell| [profile == "joint" ? 1 : 0, *cell.fetch("key")] }
    value = { "profile" => choice&.first, "choice" => choice&.last,
      "summary_sha256" => sha(File.join(directory, "seed-1337/summary.json")),
      "scope" => "Development-only profile selection: prefer an eligible three-state profile, otherwise eligible known-only binary. No acceptance opened." }
    write(path, value)
    value
  end
end
