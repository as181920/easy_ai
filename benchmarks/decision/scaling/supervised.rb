module DecisionScaling
  module_function

  def supervised_measure(root, model, tokenizer, profile, device)
    protocol = verify(root)
    controlled = (protocol.fetch("required_development") + protocol.fetch("diagnostic_development")).to_h do |name|
      [name, measure(model, tokenizer, panel(root, name, profile: profile), device)]
    end
    values = %w[development regression].to_h do |kind|
      raw = DecisionRelational.rows(File.join(root, "supervised-data", "natural-#{kind}-#{profile}.jsonl"))
      [kind, measure(model, tokenizer, raw, device)]
    end
    { "controlled" => controlled, "natural" => values.fetch("development"), "regression" => values.fetch("regression") }
  end

  def supervised_baseline(root)
    supervised_protocol(root)
    path = File.join(root, "supervised-baseline.json")
    if File.exist?(path)
      raise "Supervised baseline changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
      return JSON.parse(File.read(path))
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    device = DecisionRobust.device
    raise "Supervised baseline requires CUDA" unless device == "cuda"
    value = %w[binary joint].to_h do |profile|
      metrics = supervised_measure(root, loaded.fetch(:model), loaded.fetch(:tokenizer), profile, device)
      puts "Supervised baseline #{profile}: #{metrics.fetch('natural').fetch('by_language').transform_values { |cell| cell.fetch('accuracy') }}"
      $stdout.flush
      [profile, metrics]
    end
    loaded.fetch(:model).to("cpu")
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end

  def supervised_eligible?(value, reference, profile, protocol)
    gates = protocol.fetch("publication")
    controlled = protocol.fetch("required_development").all? do |name|
      actual = value.fetch("controlled").fetch(name)
      DecisionJudgment.eligible?(actual, profile, gates) && %w[en-US zh-CN].all? do |language|
        actual.dig("by_language", language, "balanced") >= reference.dig("controlled", name, "by_language", language, "balanced") + gates.fetch("parent_gain")
      end
    end
    controlled && %w[en-US zh-CN].all? do |language|
      actual = value.dig("natural", "by_language", language, "accuracy")
      parent = reference.dig("natural", "by_language", language, "accuracy")
      actual >= gates.fetch("natural_minimum_accuracy") && actual >= parent - gates.fetch("natural_tolerance")
    end
  end

  def supervised_key(value, protocol)
    controlled = protocol.fetch("required_development").map { |name| value.fetch("controlled").fetch(name) }
    [controlled.map { |metrics| minimum_groups(metrics) }.min, controlled.map { |metrics| metrics.fetch("worst_language_balanced") }.min,
      -controlled.sum { |metrics| metrics.fetch("nll") }]
  end

  def supervised(root, profile, seed: 1337)
    protocol = supervised_protocol(root)
    common = verify(root)
    raise ArgumentError, "Unregistered supervised profile/seed" unless protocol.fetch("profiles").include?(profile.to_s) && [1337, 2027].include?(seed)
    scale = protocol.fetch("scale")
    initial = protocol.fetch("initial_checkpoints").fetch(profile.to_s)
    if seed == 2027
      choice = supervised_selection(root)
      raise "Full recipe not eligible for confirmation" unless choice.fetch("eligible") && choice.fetch("cell") == "#{profile}-#{scale}"
      suffix = choice.fetch("controlled_steps") == 4000 ? "extended" : "screen"
      result = JSON.parse(File.read(File.join(root, "arms", "#{profile}-#{scale}-#{seed}", "#{suffix}.json")))
      unless result.fetch("known_mastery_passed")
        write(File.join(root, "confirmation.json"), { "passed" => false, "reason" => "Second-seed controlled known mastery failed", "selection_sha256" => sha(File.join(root, "supervised-selection.json")) })
        return
      end
      initial = initial.merge("checkpoint" => result.fetch("checkpoint"))
    end
    directory = File.join(root, "supervised", "#{profile}-#{scale}-#{seed}")
    raise "Preserve completed supervised experiment" if File.exist?(File.join(directory, "summary.json"))
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    config = EasyAI::Decision::Config.new(protocol.fetch("config_by_profile").fetch(profile.to_s)).with(training: { seed: seed })
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "supervised-data/train.jsonl"))
    output = File.join(directory, "choice")
    reference = supervised_baseline(root)
    if File.exist?(File.join(output, "latest.json"))
      trainer = SupervisedTrainer.resume(output, dataset: dataset, output: output)
      raise "Supervised resume config changed" unless trainer.model.config.to_h == config.to_h
    else
      loaded = EasyAI::Decision::Checkpoint.load(initial.fetch("checkpoint"))
      model = EasyAI::Decision::ChoiceModel.new(config)
      model.load_state_dict(loaded.fetch(:model).state_dict)
      write(File.join(directory, "initialization.json"), { "checkpoint" => loaded.fetch(:path), "weights_sha256" => loaded.fetch(:weights_fingerprint), "fresh_optimizer" => true })
      trainer = SupervisedTrainer.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output)
    end
    raise "Supervised experiment requires actual CUDA" unless trainer.device == "cuda"
    enforce_budget!(root, protocol.fetch("steps") - trainer.state.fetch("step"))
    probe = DecisionRelational.rows(File.join(root, "data/probe-binary.jsonl"))
    history_path = File.join(directory, "development.json")
    history = File.exist?(history_path) ? JSON.parse(File.read(history_path)).select { |cell| cell.fetch("step") <= trainer.state.fetch("step") } : []
    selected = trainer.state.fetch("supervised_selected", {})
    unless trainer.state["stop_reason"]
      trainer.train do |state, loss|
        raise "Supervised CUDA trial fell back to CPU" unless trainer.device == "cuda"
        next unless (state.fetch("step") % 250).zero?
        identities = trainer.model.named_parameters.transform_values(&:object_id)
        training = measure(trainer.model, trainer.tokenizer, probe, trainer.device)
        profiles = state.fetch("supervised_known_passed", false) ? %w[binary joint] : ["binary"]
        values = profiles.to_h { |kind| [kind, supervised_measure(root, trainer.model, trainer.tokenizer, kind, trainer.device)] }
        raise "Supervised evaluation replaced live parameters" unless identities == trainer.model.named_parameters.transform_values(&:object_id)
        if state.fetch("step") <= SupervisedTrainer::SIGNATURE.fetch("known_updates")
          mastered = DecisionRelational.known_mastery?(training, values.dig("binary", "controlled", "canonical"), common.fetch("mastery"))
          state["supervised_mastery_checks"] = mastered ? state.fetch("supervised_mastery_checks", 0) + 1 : 0
          if state.fetch("step") == SupervisedTrainer::SIGNATURE.fetch("known_updates")
            state["supervised_known_passed"] = state.fetch("supervised_mastery_checks") >= 2
            state["stop_reason"] = "Known retention failed by the fixed 500-update supervised ceiling" unless state.fetch("supervised_known_passed")
          end
        end
        values.each do |kind, value|
          next unless state.fetch("supervised_known_passed", false) && supervised_eligible?(value, reference.fetch(kind), kind, common)
          key = supervised_key(value, common)
          next if selected[kind] && (key <=> selected.fetch(kind).fetch("key")) != 1
          checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "selected-#{kind}"), model: trainer.model, tokenizer: trainer.tokenizer, training_state: state)
          selected[kind] = { "step" => state.fetch("step"), "checkpoint" => checkpoint, "key" => key, "metrics" => value }
        end
        state["supervised_selected"] = selected
        history << { "step" => state.fetch("step"), "loss" => loss, "profiles" => values, "training_probe" => training,
          "known_passed" => state.fetch("supervised_known_passed", false), "mastery_checks" => state.fetch("supervised_mastery_checks", 0), "device" => trainer.device }
        write(history_path, history)
        puts "Supervised #{profile}/#{scale}/#{seed} step=#{state['step']} known=#{state['supervised_known_passed']} eligible=#{selected.keys}"
        $stdout.flush
      end
    end
    summary = { "step" => trainer.state.fetch("step"), "known_passed" => trainer.state.fetch("supervised_known_passed", false), "selected" => selected,
      "stop_reason" => trainer.state["stop_reason"], "checkpoint" => trainer.last_checkpoint, "device" => trainer.device,
      "examples_seen" => trainer.state.fetch("examples_seen"), "input_tokens_seen" => trainer.state.dig("coverage", "input_tokens"),
      "controlled_steps" => initial.fetch("steps"), "scope" => "Observed development only; no calibration or acceptance result." }
    write(File.join(directory, "summary.json"), summary)
    if seed == 2027
      choice = supervised_selection(root)
      write(File.join(root, "confirmation.json"), { "passed" => selected.key?(choice.fetch("profile")), "profile" => choice.fetch("profile"),
        "selection_sha256" => sha(File.join(root, "supervised-selection.json")), "summary_sha256" => sha(File.join(directory, "summary.json")),
        "scope" => "Second seed repeats controlled initialization/training and the same supervised curriculum; same required development profile, no acceptance opened." })
    end
    puts "Completed supervised #{profile}/#{scale}/#{seed} at #{summary.fetch('step')}"
  end
end
