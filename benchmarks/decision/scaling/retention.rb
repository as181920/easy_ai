module DecisionScaling
  module_function

  def prepare_retained(root)
    original = supervised_protocol(root)
    decision = supervised_selection(root)
    raise "Correction requires a terminal negative primary" if decision.fetch("eligible")
    raise "Unregistered correction cell" unless original.fetch("profiles") == ["base"] && original.fetch("scale") == 32
    path = File.join(root, "retained-protocol.json")
    raise "Retention correction already frozen" if File.exist?(path)
    summary_path = File.join(root, "supervised/base-32-1337/summary.json")
    summary = JSON.parse(File.read(summary_path))
    history_path = File.join(root, "supervised/base-32-1337/development.json")
    history = JSON.parse(File.read(history_path))
    unless summary.fetch("step") == 500 && !summary.fetch("known_passed") && history.last.fetch("mastery_checks") == 1 && history.map { |cell| cell.fetch("step") } == [250, 500]
      raise "Correction requires the measured two-check retention-window failure"
    end
    enforce_budget!(root, 2500)
    checkpoint = summary.fetch("checkpoint")
    files = %w[weights.pt optimizer.pt metadata.json tokenizer.json manifest.json]
    source = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort
    source.each do |file|
      destination = File.join(root, "retained-source-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
    end
    value = { "original_protocol_sha256" => sha(File.join(root, "supervised-protocol.json")),
      "original_summary_sha256" => sha(summary_path), "original_history_sha256" => sha(history_path),
      "original_selection_sha256" => sha(File.join(root, "supervised-selection.json")),
      "fork_checkpoint" => checkpoint, "fork_files_sha256" => files.to_h { |name| [name, sha(File.join(checkpoint, name))] },
      "sampler" => RetainedTrainer::RETENTION_SIGNATURE, "inherited_steps" => 500, "total_steps" => 3000,
      "config" => original.fetch("config_by_profile").fetch("base"), "plan_sha256" => sha("docs/decision/v06-retention-plan.md"),
      "source_sha256" => source.to_h { |file| [file, sha(file)] },
      "scope" => "One corrective fork, preserving the original terminal failure. Inherit its first 500 model/optimizer/data steps; do not copy the trace as new updates. Same data/gates; stronger known replay and maximum known window 1000." }
    FileUtils.cp("docs/decision/v06-retention-plan.md", File.join(root, "retained-plan-frozen.md"))
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts "Frozen corrective retention fork: 500 inherited, at most 2500 new updates"
  end

  def retained_protocol(root)
    supervised_protocol(root)
    path = File.join(root, "retained-protocol.json")
    raise "Retention protocol changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    value = JSON.parse(File.read(path))
    { "supervised-protocol.json" => "original_protocol_sha256", "supervised-selection.json" => "original_selection_sha256",
      "supervised/base-32-1337/summary.json" => "original_summary_sha256", "supervised/base-32-1337/development.json" => "original_history_sha256" }.each do |file, key|
      raise "Original retention evidence changed" unless sha(File.join(root, file)) == value.fetch(key)
    end
    value.fetch("fork_files_sha256").each { |name, digest| raise "Fork checkpoint changed" unless sha(File.join(value.fetch("fork_checkpoint"), name)) == digest }
    value
  end

  def retained(root)
    protocol = retained_protocol(root)
    common = verify(root)
    directory = File.join(root, "supervised/retained-base-32-1337")
    raise "Preserve completed retention correction" if File.exist?(File.join(directory, "summary.json"))
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "supervised-data/train.jsonl"))
    output = File.join(directory, "choice")
    resumed = File.exist?(File.join(output, "latest.json"))
    source = resumed ? output : protocol.fetch("fork_checkpoint")
    trainer = RetainedTrainer.resume(source, dataset: dataset, output: output, fork_from_original: !resumed)
    raise "Retention correction config changed" unless trainer.model.config.to_h == protocol.fetch("config")
    raise "Correction requires actual CUDA" unless trainer.device == "cuda"
    enforce_budget!(root, protocol.fetch("total_steps") - trainer.state.fetch("step"))
    history_path = File.join(directory, "development.json")
    history = if File.exist?(history_path)
      JSON.parse(File.read(history_path)).select { |cell| cell.fetch("step") <= trainer.state.fetch("step") }
              else
      JSON.parse(File.read(File.join(root, "supervised/base-32-1337/development.json"))).map { |cell| cell.merge("inherited" => true) }
              end
    write(history_path, history)
    reference = supervised_baseline(root)
    probe = DecisionRelational.rows(File.join(root, "data/probe-binary.jsonl"))
    selected = trainer.state.fetch("retained_selected", {})
    remaining = trainer.state["stop_reason"] ? 0 : protocol.fetch("total_steps") - trainer.state.fetch("step")
    trainer.train(steps: trainer.state.fetch("step") + remaining) do |state, loss|
      raise "Corrective CUDA trial fell back to CPU" unless trainer.device == "cuda"
      next unless (state.fetch("step") % 250).zero?
      identities = trainer.model.named_parameters.transform_values(&:object_id)
      training = measure(trainer.model, trainer.tokenizer, probe, trainer.device)
      profiles = state.fetch("supervised_known_passed", false) ? %w[binary joint] : ["binary"]
      values = profiles.to_h { |profile| [profile, supervised_measure(root, trainer.model, trainer.tokenizer, profile, trainer.device)] }
      raise "Correction evaluation replaced parameters" unless identities == trainer.model.named_parameters.transform_values(&:object_id)
      unless state.fetch("supervised_known_passed", false)
        mastered = DecisionRelational.known_mastery?(training, values.dig("binary", "controlled", "canonical"), common.fetch("mastery"))
        state["retained_mastery_checks"] = mastered ? state.fetch("retained_mastery_checks", 0) + 1 : 0
        if state.fetch("retained_mastery_checks") >= 2
          state["supervised_known_passed"] = true
          state["retained_joint_started_at"] = state.fetch("step")
        elsif state.fetch("step") >= RetainedTrainer::RETENTION_SIGNATURE.fetch("maximum_known_updates")
          state["stop_reason"] = "Corrective known retention failed by the fixed 1000-step ceiling"
        end
      end
      values.each do |profile, value|
        next unless state.fetch("supervised_known_passed", false) && supervised_eligible?(value, reference.fetch(profile), profile, common)
        key = supervised_key(value, common)
        next if selected[profile] && (key <=> selected.fetch(profile).fetch("key")) != 1
        checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "selected-#{profile}"), model: trainer.model, tokenizer: trainer.tokenizer, training_state: state)
        selected[profile] = { "step" => state.fetch("step"), "checkpoint" => checkpoint, "key" => key, "metrics" => value }
      end
      state["retained_selected"] = selected
      history << { "step" => state.fetch("step"), "loss" => loss, "profiles" => values, "training_probe" => training, "inherited" => false,
        "known_passed" => state.fetch("supervised_known_passed", false), "mastery_checks" => state.fetch("retained_mastery_checks", 0), "device" => trainer.device }
      write(history_path, history)
      puts "Retention step=#{state['step']} known=#{state['supervised_known_passed']} eligible=#{selected.keys}"
      $stdout.flush
    end
    write(File.join(directory, "summary.json"), { "step" => trainer.state.fetch("step"), "known_passed" => trainer.state.fetch("supervised_known_passed", false),
      "selected" => selected, "stop_reason" => trainer.state["stop_reason"], "checkpoint" => trainer.last_checkpoint, "device" => trainer.device,
      "inherited_steps" => 500, "actual_new_trace_updates" => File.foreach(File.join(output, "training.jsonl")).count,
      "known_updates" => trainer.state.fetch("retained_joint_started_at", trainer.state.fetch("step")),
      "unknown_updates" => trainer.state["retained_joint_started_at"] ? trainer.state.fetch("step") - trainer.state.fetch("retained_joint_started_at") : 0,
      "examples_seen" => trainer.state.fetch("examples_seen"), "input_tokens_seen" => trainer.state.dig("coverage", "input_tokens"),
      "scope" => "Explicit corrective fork. Recipe step/exposure totals include 500 inherited updates; actual_new_trace_updates and global ledger count only newly executed updates. No acceptance opened." })
  end

  def select_retained(root)
    retained_protocol(root)
    path = File.join(root, "retained-selection.json")
    raise "Retention selection already frozen" if File.exist?(path)
    summary_path = File.join(root, "supervised/retained-base-32-1337/summary.json")
    summary = JSON.parse(File.read(summary_path))
    unless summary.fetch("step") == 3000 || (summary.fetch("step") == 1000 && !summary.fetch("known_passed"))
      raise "Retention correction incomplete"
    end
    winner = summary.fetch("selected").max_by { |profile, cell| [profile == "joint" ? 1 : 0, *cell.fetch("key")] }
    value = { "eligible" => !winner.nil?, "profile" => winner&.first, "selected" => winner&.last,
      "summary_sha256" => sha(summary_path), "protocol_sha256" => sha(File.join(root, "retained-protocol.json")),
      "scope" => "A positive corrected recipe still requires full own-parent second-seed confirmation before calibration/acceptance. A negative selection closes this finite corrective branch." }
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end
end
