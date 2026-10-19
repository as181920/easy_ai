module DecisionRelational
  module_function

  def fit(root, arm, seed, selection: "default")
    fit_selection_path(root, selection)
    protocol = verify(root)
    raw_config, initialization = recipe_config(root, protocol, arm)
    raise ArgumentError, "Unsupported seed" unless [1337, 2027].include?(seed)
    check_confirmation!(root, arm, selection: selection) if seed == 2027
    directory = File.join(root, "fit-#{arm}-#{seed}")
    raise "Completed experiment exists" if File.exist?(File.join(directory, "result.json"))
    config = EasyAI::Decision::Config.new(raw_config).with(training: { seed: seed })
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/fit.jsonl"))
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    output = File.join(directory, "choice")
    trainer = if File.exist?(File.join(output, "latest.json"))
                restored = EasyAI::Decision::CandidateTrainer.resume(output, dataset: dataset, output: output)
                raise "Resume configuration changed" unless restored.model.config.to_h == config.to_h
                restored
              else
                loaded = EasyAI::Decision::Checkpoint.load(PARENT)
                model, manifest = if initialization == "scratch"
                                    Torch.manual_seed(seed)
                                    fresh = EasyAI::Decision::ChoiceModel.new(config)
                                    [fresh, { "copied" => [], "new" => fresh.state_dict.keys.sort, "parameters" => fresh.parameter_count, "initialization" => "scratch" }]
                                  else
                                    EasyAI::Decision::WeightTransfer.apply(parent: loaded, config: config, tokenizer: loaded.fetch(:tokenizer), seed: seed)
                                  end
                write(File.join(directory, "transfer.json"), manifest)
                EasyAI::Decision::CandidateTrainer.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output)
              end
    enforce_update_budget!(root, protocol.dig("fit", "maximum_steps") - trainer.state.fetch("step"))
    execute_fit(root, directory, trainer, protocol)
  end

  def execute_fit(root, directory, trainer, protocol)
    raw = rows(File.join(root, "data/fit.jsonl"))
    path = File.join(directory, "measurements.json")
    history = File.exist?(path) ? JSON.parse(File.read(path)).select { |cell| cell.fetch("step") <= trainer.state.fetch("step") } : []
    stable = history.reverse.take_while { |cell| mastered?(cell, protocol.fetch("fit")) }.size
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    (100..protocol.dig("fit", "maximum_steps")).step(100) do |step|
      next if step <= trainer.state.fetch("step")
      trainer.train(steps: step)
      identities = trainer.model.named_parameters.transform_values(&:object_id)
      value = DecisionJudgment.measure(trainer.model, trainer.tokenizer, raw, trainer.device)
      raise "Evaluation replaced live parameters" unless identities == trainer.model.named_parameters.transform_values(&:object_id)
      value.merge!("step" => step, "device" => trainer.device, "elapsed_seconds_session" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - started,
        "training_loss" => trainer.state.fetch("last_train_loss"), "microbatch" => trainer.state.fetch("microbatch"),
        "gradient_accumulation" => trainer.state.fetch("gradient_accumulation"), "examples_seen" => trainer.state.fetch("examples_seen"),
        "input_tokens_seen" => trainer.state.dig("coverage", "input_tokens"),
        "process_mib" => trainer.device == "cuda" ? EasyAI::Runtime::DevicePolicy.new(requested: "cuda").process_memory_mib : nil)
      stable = mastered?(value, protocol.fetch("fit")) ? stable + 1 : 0
      history << value
      write(path, history)
      puts "#{File.basename(directory)} step=#{step} loss=#{value['training_loss'].round(5)} metrics=#{value.fetch('by_language').transform_values { |cell| cell.slice('balanced', 'actor_binding_all_correct', 'event_binding_all_correct') }} device=#{trainer.device}"
      $stdout.flush
      break if stable >= protocol.dig("fit", "stable_checks")
    end
    development = DecisionJudgment.panel(root, "validation", "binary")
    value = DecisionJudgment.measure(trainer.model, trainer.tokenizer, development, trainer.device)
    write(File.join(directory, "result.json"), { "passed" => stable >= protocol.dig("fit", "stable_checks"),
      "step" => trainer.state.fetch("step"), "checkpoint" => trainer.last_checkpoint, "measurements" => history,
      "observed_development" => value, "device" => trainer.device,
      "scope" => "Fitting diagnostic and previously observed development only; no fresh acceptance, no promoted model" })
  end
end
