module DecisionRobust
  module_function

  def device
    EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
  end

  def baseline(root)
    protocol = verify(root)
    raise "Baselines already frozen" if File.exist?(File.join(root, "baseline.json"))
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    rows = raw_rows(File.join(root, "data/validation.jsonl"))
    value = measure(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, device)
    probe = raw_rows(File.join(root, "data/fit.jsonl")).select { |row| row["world"] }.first(52)
    diagnostics = DecisionFactual.diagnostics(loaded.fetch(:model), loaded.fetch(:tokenizer), probe, device)
    write(File.join(root, "baseline.json"), { "validation" => value, "diagnostics" => diagnostics,
      "parent_sha256" => protocol.fetch("parent_sha256"), "scope" => "Delivered v0.1 on shared development material; no acceptance opened" })
    write(File.join(root, "baseline-snapshot.json"), { "baseline_sha256" => sha(File.join(root, "baseline.json")),
      "protocol_sha256" => sha(File.join(root, "protocol.json")) })
    puts "v0.1 development macro=#{value.fetch('factual_macro_accuracy')}"
  end

  def trainer_for(root, arm, seed, fitting: false)
    raise ArgumentError, "Unknown arm" unless ARMS.include?(arm)
    protocol = verify(root)
    directory = File.join(root, fitting ? "fit" : "#{arm}-#{seed}")
    raise "Completed run exists" if File.exist?(File.join(directory, "summary.json"))
    cfg = EasyAI::Decision::Config.new(protocol.fetch("config")).with(training: { seed: seed })
    cfg = cfg.with(model: { dropout: 0.0 }, training: { learning_rate: 0.0003, warmup_steps: 0 }) if fitting
    path = File.join(root, fitting ? "data/fit.jsonl" : "data/#{arm}-train.jsonl")
    dataset = EasyAI::Decision::Data::Dataset.new(path)
    klass = arm == "candidate" ? EasyAI::Decision::RobustTrainer : EasyAI::Decision::FactualTrainer
    output = File.join(directory, "choice")
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    if File.exist?(File.join(output, "latest.json"))
      trainer = klass.resume(output, dataset: dataset, output: output)
      raise "Resume config changed" unless trainer.model.config.to_h == cfg.to_h
      return trainer
    end
    parent = EasyAI::Decision::Checkpoint.load(PARENT)
    model = EasyAI::Decision::ChoiceModel.new(cfg)
    model.load_state_dict(parent.fetch(:model).state_dict)
    klass.new(model: model, tokenizer: parent.fetch(:tokenizer), dataset: dataset, output: output)
  end

  def fit(root)
    protocol = verify(root)
    raise "Fitting result already frozen" if File.exist?(File.join(root, "fit/result.json"))
    trainer = trainer_for(root, "candidate", 1337, fitting: true)
    rows = raw_rows(File.join(root, "data/fit.jsonl"))
    development = raw_rows(File.join(root, "data/validation.jsonl"))
    history_path = File.join(root, "fit/measurements.json")
    measurements = File.exist?(history_path) ? JSON.parse(File.read(history_path)) : []
    stable = 0
    limit = protocol.fetch("fit")
    (100..limit.fetch("maximum_steps")).step(100) do |budget|
      next if budget <= trainer.state.fetch("step")
      trainer.train(steps: budget)
      result = measure(trainer.model, trainer.tokenizer, rows, trainer.device).merge("step" => trainer.state.fetch("step"))
      measurements << result
      write(history_path, measurements)
      complete = result.fetch("robust_by_language").values.sum { |cell| cell.fetch("world_all_correct") }.fdiv(2)
      stable = result.fetch("accuracy") >= limit.fetch("accuracy") && complete >= limit.fetch("groups") ? stable + 1 : 0
      puts "Fit step=#{budget} accuracy=#{result['accuracy'].round(4)} worlds=#{complete.round(4)}"
      break if stable >= limit.fetch("stable_checks")
    end
    result = { "passed" => stable >= limit.fetch("stable_checks"), "measurements" => measurements,
      "unseen_development" => measure(trainer.model, trainer.tokenizer, development, trainer.device),
      "scope" => "Fitting uses training examples only; unseen development expressions measured separately; weights not used as pilot parent" }
    write(File.join(root, "fit/result.json"), result)
    raise "Fitting check failed; diagnose before pilot" unless result.fetch("passed")
  end

  def eligible?(metrics, reference, gates)
    %w[en-US zh-CN].all? do |language|
      cell = metrics.fetch("robust_by_language").fetch(language)
      metrics.fetch("routing_by_language").fetch(language).fetch("accuracy") >= reference.fetch("routing_by_language").fetch(language).fetch("accuracy") - gates.fetch("routing_tolerance") &&
        cell.fetch("known_accuracy") >= gates.fetch("minimum_known_accuracy") && cell.fetch("unknown_accuracy") >= gates.fetch("minimum_unknown_accuracy") &&
        cell.fetch("binding_all_correct") >= gates.fetch("minimum_binding_group_accuracy") &&
        cell.fetch("mixed_binding_all_correct") >= gates.fetch("minimum_mixed_binding_group_accuracy") && cell.fetch("wording_accuracy") >= gates.fetch("minimum_wording_accuracy")
    end
  end

  def train(root, arm, seed)
    protocol = verify(root)
    raise "Fitting has not passed" unless JSON.parse(File.read(File.join(root, "fit/result.json"))).fetch("passed")
    audit_control_selection(root) if arm == "candidate" && File.exist?(File.join(root, "control-1337/summary.json"))
    trainer = trainer_for(root, arm, seed)
    directory = File.join(root, "#{arm}-#{seed}")
    rows = raw_rows(File.join(root, "data/validation.jsonl"))
    probe = raw_rows(File.join(root, "data/fit.jsonl"))
    reference = JSON.parse(File.read(File.join(root, "baseline.json"))).fetch("validation")
    initial = EasyAI::Decision::Checkpoint.load(PARENT).fetch(:model).named_parameters.fetch("encoder.embedding.weight").detach.cpu.clone
    trace_path = File.join(directory, "validation.json")
    trace = File.exist?(trace_path) ? JSON.parse(File.read(trace_path)) : []
    trace.reject! { |row| row.fetch("step") > trainer.state.fetch("step") }
    selection_path = File.join(directory, "selection.json")
    best = File.exist?(selection_path) ? JSON.parse(File.read(selection_path)) : nil
    best = nil if best && best.fetch("metrics").fetch("step") > trainer.state.fetch("step")
    trainer.train do |state, loss|
      puts "#{arm}/#{seed} step=#{state['step']} loss=#{loss.round(4)} device=#{trainer.device}" if (state.fetch("step") % 20).zero?
      next unless (state.fetch("step") % 100).zero?
      metrics = measure(trainer.model, trainer.tokenizer, rows, trainer.device)
      metrics.merge!("step" => state.fetch("step"), "device" => trainer.device,
        "fixed_train_probe" => measure(trainer.model, trainer.tokenizer, probe, trainer.device),
        "probe_scope" => "Candidate-training fitting set; generated worlds are not in control training",
        "embedding_max_change" => (trainer.model.named_parameters.fetch("encoder.embedding.weight").detach.cpu - initial).abs.max.item,
        "gpu_process_mib" => trainer.device == "cuda" ? EasyAI::Runtime::DevicePolicy.new(requested: "cuda").process_memory_mib : nil)
      metrics["eligible"] = eligible?(metrics, reference, protocol.fetch("selection"))
      key = [metrics.fetch("factual_macro_accuracy"), -metrics.fetch("factual_nll")]
      if metrics.fetch("eligible") && (best.nil? || (key <=> best.fetch("key")) == 1)
        checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "selected"), model: trainer.model, tokenizer: trainer.tokenizer, training_state: state)
        best = { "key" => key, "checkpoint" => checkpoint, "metrics" => metrics }
        write(selection_path, best)
      end
      trace.reject! { |row| row.fetch("step") == state.fetch("step") }
      trace << metrics
      write(trace_path, trace)
      puts "#{arm}/#{seed} factual=#{metrics['factual_macro_accuracy'].round(4)} eligible=#{metrics['eligible']} cells=#{metrics['robust_by_language'].transform_values { |cell| cell.slice('known_accuracy', 'unknown_accuracy', 'binding_all_correct', 'wording_accuracy') }}"
    end
    write(File.join(directory, "summary.json"), { "step" => trainer.state.fetch("step"), "selected" => best,
      "last_checkpoint" => trainer.last_checkpoint, "device" => trainer.device, "examples_seen" => trainer.state.fetch("examples_seen"),
      "coverage" => trainer.state.fetch("coverage") })
  end

  def audit_control_selection(root)
    directory = File.join(root, "control-1337")
    audit_path = File.join(directory, "selection-audit.json")
    return if File.exist?(audit_path)
    protocol = verify(root)
    summary = JSON.parse(File.read(File.join(directory, "summary.json")))
    trace = JSON.parse(File.read(File.join(directory, "validation.json")))
    reference = JSON.parse(File.read(File.join(root, "baseline.json"))).fetch("validation")
    rows = raw_rows(File.join(root, "data/validation.jsonl"))
    best = nil
    rechecked = []
    # The added mixed-truth guard only narrows eligibility. Previously ineligible points stay ineligible.
    trace.select { |row| row.fetch("eligible") }.each do |previous|
      path = Dir.glob(File.join(directory, "choice/checkpoints/step-#{format('%08d', previous.fetch('step'))}-*/metadata.json")).max
      raise "Control checkpoint missing for audit" unless path
      loaded = EasyAI::Decision::Checkpoint.load(File.dirname(path))
      metrics = measure(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, device).merge("step" => previous.fetch("step"))
      allowed = eligible?(metrics, reference, protocol.fetch("selection"))
      key = [metrics.fetch("factual_macro_accuracy"), -metrics.fetch("factual_nll")]
      rechecked << { "step" => previous.fetch("step"), "eligible" => allowed, "metrics" => metrics }
      best = { "key" => key, "checkpoint" => loaded.fetch(:path), "metrics" => metrics } if allowed && (best.nil? || (key <=> best.fetch("key")) == 1)
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
    end
    summary["original_selected"] = summary.fetch("selected")
    summary["selected"] = best
    write(File.join(directory, "summary.json"), summary)
    write(audit_path, { "rechecked" => rechecked, "selected" => best,
      "scope" => "Mixed-truth actor guard clarified before candidate training and final evaluation; no updates or test-driven tuning" })
  end

  def confirm(root)
    protocol = verify(root)
    raise "Confirmation already frozen" if File.exist?(File.join(root, "confirmation.json"))
    summaries = ARMS.to_h { |arm| [arm, JSON.parse(File.read(File.join(root, "#{arm}-1337/summary.json")))] }
    raise "Pilot incomplete" unless summaries.values.all? { |result| result.fetch("step") == STEPS }
    reference = JSON.parse(File.read(File.join(root, "baseline.json"))).fetch("validation").fetch("factual_macro_accuracy")
    candidate = summaries.fetch("candidate").fetch("selected")
    control = summaries.fetch("control").fetch("selected")
    # Compare against the control's best factual development result even if it fails slice guards.
    control_macro = JSON.parse(File.read(File.join(root, "control-1337/validation.json"))).map { |row| row.fetch("factual_macro_accuracy") }.max
    rules = protocol.fetch("confirmation")
    passed = candidate && candidate.fetch("key").first >= reference + rules.fetch("minimum_parent_gain") &&
      candidate.fetch("key").first >= control_macro + rules.fetch("minimum_control_gain")
    decision = { "winner" => passed ? "candidate" : nil, "confirmation_required" => !!passed,
      "parent_macro" => reference, "control_best_macro" => control_macro, "candidate_selected" => candidate&.fetch("key"),
      "control_selected" => control&.fetch("key"), "criterion" => rules, "scope" => "Development-only decision; no acceptance opened" }
    write(File.join(root, "confirmation.json"), decision)
    puts JSON.pretty_generate(decision)
    train(root, "candidate", rules.fetch("seed")) if passed
  end
end
