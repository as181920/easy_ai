module DecisionScaling
  module_function

  def panel(root, name, profile: "binary")
    file = name.to_s == "canonical" ? "validation" : "validation-#{name}"
    rows = DecisionRelational.rows(File.join(root, "data", "#{file}.jsonl"))
    rows.select { |row| row.dig("world", "profile") == profile.to_s }
  end

  def measure(model, tokenizer, raw, device)
    DecisionJudgment.measure(model, tokenizer, raw, device, gc_interval: 8)
  end

  def baseline(root)
    protocol = verify(root)
    path = File.join(root, "baseline.json")
    if File.exist?(path)
      raise "Scaling baseline changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
      return JSON.parse(File.read(path))
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    device = DecisionRobust.device
    values = (protocol.fetch("required_development") + protocol.fetch("diagnostic_development")).to_h do |name|
      metrics = measure(loaded.fetch(:model), loaded.fetch(:tokenizer), panel(root, name), device)
      puts "Baseline #{name}: #{metrics.fetch('worst_language_balanced')}"
      $stdout.flush
      [name, metrics]
    end
    loaded.fetch(:model).to("cpu")
    write(path, values)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    values
  end

  def controlled_eligible?(values, reference, protocol)
    protocol.fetch("required_development").all? do |name|
      value = values.fetch(name)
      DecisionJudgment.eligible?(value, "binary", protocol.fetch("publication")) && %w[en-US zh-CN].all? do |language|
        value.dig("by_language", language, "balanced") >= reference.dig(name, "by_language", language, "balanced") + protocol.dig("publication", "parent_gain")
      end
    end
  end

  def train(root, profile, scale, seed: 1337, limit: 1500)
    protocol = verify(root)
    midpoint = scale == 128 && profile.to_s == "base" && extension_decision(root).fetch("midpoint")
    raise ArgumentError, "Unregistered factorial cell" unless protocol.fetch("profiles").include?(profile.to_s) && ([32, 512].include?(scale) || midpoint)
    raise ArgumentError, "Unregistered seed or budget" unless [1337, 2027].include?(seed) && [1500, 4000].include?(limit)
    if limit == 4000 && seed == 1337
      decision = extension_decision(root)
      raise "Cell not authorized for extension" unless decision.fetch("extend").include?("#{profile}-#{scale}") && seed == 1337
    end
    if seed == 2027
      final = JSON.parse(File.read(File.join(root, "supervised-selection.json")))
      raise "Confirmation requires a frozen eligible full recipe" unless final.fetch("eligible") && final.fetch("cell") == "#{profile}-#{scale}" && final.fetch("controlled_steps") == limit
    end
    directory = File.join(root, "arms", "#{profile}-#{scale}-#{seed}")
    result_path = File.join(directory, limit == 1500 ? "screen.json" : "extended.json")
    raise "Preserve completed scale experiment" if File.exist?(result_path)
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data", "train-#{scale}-binary.jsonl"))
    output = File.join(directory, "choice")
    if File.exist?(File.join(output, "latest.json"))
      trainer = QuadTrainer.resume(output, dataset: dataset, output: output)
      expected = EasyAI::Decision::Config.new(protocol.fetch("config_by_profile").fetch(profile.to_s)).with(training: { seed: seed })
      raise "Scaling resume config changed" unless trainer.model.config.to_h == expected.to_h
    else
      model, tokenizer, transfer = initialize_model(profile, microbatch: protocol.dig("hardware", "controlled_microbatch"), seed: seed)
      write(File.join(directory, "initialization.json"), transfer)
      trainer = QuadTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: output)
    end
    raise "Scale experiment requires actual CUDA" unless trainer.device == "cuda"
    enforce_budget!(root, limit - trainer.state.fetch("step"))
    reference = baseline(root)
    probe = DecisionRelational.rows(File.join(root, "data/probe-binary.jsonl"))
    history_path = File.join(directory, "development.json")
    history = File.exist?(history_path) ? JSON.parse(File.read(history_path)).select { |cell| cell.fetch("step") <= trainer.state.fetch("step") } : []
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    trainer.train(steps: limit) do |state, loss|
      raise "CUDA trial fell back to CPU" unless trainer.device == "cuda"
      next unless (state.fetch("step") % protocol.fetch("evaluation_every")).zero?
      identities = trainer.model.named_parameters.transform_values(&:object_id)
      training = measure(trainer.model, trainer.tokenizer, probe, trainer.device)
      value = measure(trainer.model, trainer.tokenizer, panel(root, "canonical"), trainer.device)
      raise "Scale evaluation replaced live parameters" unless identities == trainer.model.named_parameters.transform_values(&:object_id)
      mastered = DecisionRelational.known_mastery?(training, value, protocol.fetch("mastery"))
      stable = mastered ? history.reverse.take_while { |cell| cell.fetch("mastered") }.size + 1 : 0
      history << { "step" => state.fetch("step"), "loss" => loss, "training_probe" => training, "canonical" => value,
        "mastered" => mastered, "stable_mastery_checks" => stable, "device" => trainer.device,
        "examples_seen" => state.fetch("examples_seen"), "input_tokens_seen" => state.dig("coverage", "input_tokens"),
        "sampled_process_mib" => EasyAI::Runtime::DevicePolicy.new(requested: :cuda).process_memory_mib,
        "session_seconds" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - started }
      write(history_path, history)
      puts "Scale #{profile}/#{scale}/#{seed} step=#{state['step']} dev=#{value.fetch('worst_language_balanced')} train=#{training.fetch('worst_language_balanced')} mastery=#{stable}"
      $stdout.flush
    end
    values = { "canonical" => history.last.fetch("canonical") }
    (protocol.fetch("required_development") + protocol.fetch("diagnostic_development") - ["canonical"]).each do |name|
      values[name] = measure(trainer.model, trainer.tokenizer, panel(root, name), trainer.device)
    end
    write(result_path, { "step" => trainer.state.fetch("step"), "checkpoint" => trainer.last_checkpoint,
      "known_mastery_passed" => history.any? { |cell| cell.fetch("stable_mastery_checks") >= 2 },
      "controlled_eligible" => controlled_eligible?(values, reference, protocol), "development" => values,
      "final_training_probe" => history.last.fetch("training_probe"), "parameters" => trainer.model.parameter_count,
      "device" => trainer.device, "examples_seen" => trainer.state.fetch("examples_seen"), "input_tokens_seen" => trainer.state.dig("coverage", "input_tokens"),
      "scope" => "Controlled-only scale screen; no natural generalization, calibration, acceptance or delivery is established." })
    puts "Completed scale #{profile}/#{scale} at #{limit}"
  end
end
