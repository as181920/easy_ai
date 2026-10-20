module DecisionBindingGeneralization
  module_function

  def preflight(root)
    verify(root)
    path = File.join(root, "feasibility.json")
    raise "Preserve completed feasibility" if File.exist?(path)
    budget!(root, 12)
    records = VARIANTS.to_h do |variant|
      output = File.join(root, "preflight", variant)
      trial = trainer(root, variant, 1337, output)
      raise "v07 requires CUDA" unless trial.device == "cuda"
      collator = trial.instance_variable_get(:@collator)
      original_call = collator.method(:call)
      collator.define_singleton_method(:call) do |examples, **options|
        batch = original_call.call(examples, **options)
        size = examples.size
        batch[:state_ids] = Torch.full([size, 64], 5, dtype: :int64, device: options.fetch(:device))
        batch[:state_mask] = Torch.ones([size, 64], dtype: :bool, device: options.fetch(:device))
        batch[:option_ids] = Torch.full([size, 2, 64], 5, dtype: :int64, device: options.fetch(:device))
        batch[:option_mask] = Torch.ones([size, 2, 64], dtype: :bool, device: options.fetch(:device))
        batch[:answer_mask] = batch[:option_mask]
        @input_token_counts = Array.new(size, 192)
        batch
      end
      memory = []
      trial.train(steps: 3) do
        raise "Feasibility fell back to CPU" unless trial.device == "cuda"
        memory << EasyAI::Runtime::DevicePolicy.new(requested: :cuda).process_memory_mib
      end
      record = { "passed" => true, "device" => trial.device, "steps" => 3, "state_tokens" => 64, "option_tokens" => 64,
        "microbatch" => 16, "effective_batch" => 32, "sampled_process_mib" => memory, "parameters" => trial.model.parameter_count }
      puts "v07 feasibility #{variant}: #{memory.max} MiB"
      $stdout.flush
      trial.model.to("cpu")
      trial = nil
      GC.start
      [variant, record]
    end
    write(path, records)
    write(path + ".sha256.json", { "sha256" => sha(path) })
  end

  def train(root, variant, seed, steps)
    protocol = verify(root)
    variant = variant.to_s
    raise ArgumentError, "Unregistered variant/seed/endpoint" unless VARIANTS.include?(variant) && [1337, 2027].include?(seed) && [6000, 12000].include?(steps)
    path = File.join(root, "feasibility.json")
    raise "Feasibility changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    reference_confirmation = seed == 2027 && variant == "ce" && steps == 6000 && File.exist?(File.join(root, "exposure-confirmation/amendment.json"))
    if (seed == 2027 || steps == 12000) && !reference_confirmation
      decision = decision(root)
      if seed == 2027
        raise "No improving recipe to confirm" unless decision.fetch("improving") && ["ce", decision.fetch("winner")].include?(variant) && steps == 6000
      else
        raise "Extension not authorized" unless variant == "ce" && decision.fetch("extend_ce")
      end
    end
    directory = File.join(root, "arms", "#{variant}-#{seed}")
    FileUtils.mkdir_p(directory)
    result_path = File.join(directory, "result-#{steps}.json")
    raise "Preserve completed v07 endpoint" if File.exist?(result_path)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    output = File.join(directory, "choice")
    trial = trainer(root, variant, seed, output, resume: File.exist?(File.join(output, "latest.json")))
    raise "v07 requires actual CUDA" unless trial.device == "cuda"
    budget!(root, steps - trial.state.fetch("step"))
    history_path = File.join(directory, "development.json")
    history = File.exist?(history_path) ? JSON.parse(File.read(history_path)).select { |cell| cell.fetch("step") <= trial.state.fetch("step") } : []
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    trial.train(steps: steps) do |state, loss|
      raise "v07 trial fell back to CPU" unless trial.device == "cuda"
      unless trial.instance_variable_get(:@microbatch) == 16 && trial.instance_variable_get(:@accumulation) == 2
        raise "v07 comparison batch layout changed"
      end
      if (state.fetch("step") % 100).zero?
        puts "v07 #{variant}/#{seed} step=#{state.fetch('step')} loss=#{loss.round(5)}"
        $stdout.flush
      end
      next unless (state.fetch("step") % 500).zero?
      ids = trial.model.named_parameters.transform_values(&:object_id)
      training = DecisionScaling.measure(trial.model, trial.tokenizer, raw(root, "probe"), trial.device)
      canonical = DecisionScaling.measure(trial.model, trial.tokenizer, raw(root, "canonical"), trial.device)
      raise "v07 evaluation replaced parameters" unless ids == trial.model.named_parameters.transform_values(&:object_id)
      mastered = DecisionRelational.known_mastery?(training, canonical, protocol.fetch("mastery"))
      state["v07_mastery_checks"] = mastered ? state.fetch("v07_mastery_checks", 0) + 1 : 0
      history << { "step" => state.fetch("step"), "loss" => loss, "training_probe" => training, "canonical" => canonical,
        "mastery_checks" => state.fetch("v07_mastery_checks"), "device" => trial.device,
        "sampled_process_mib" => EasyAI::Runtime::DevicePolicy.new(requested: :cuda).process_memory_mib }
      write(history_path, history)
      puts "v07 #{variant}/#{seed}: train=#{training.fetch('worst_language_balanced').round(4)} groups=#{DecisionScaling.minimum_groups(training).round(4)} dev=#{canonical.fetch('worst_language_balanced').round(4)}"
      $stdout.flush
    end
    measured = { "canonical" => history.last.fetch("canonical") }
    %w[seen_combinations novel_actor novel_event].each { |name| measured[name] = DecisionScaling.measure(trial.model, trial.tokenizer, raw(root, name), trial.device) }
    full = DecisionScaling.measure(trial.model, trial.tokenizer, raw(root, "train"), trial.device)
    metadata = trial.state.fetch("coverage")
    value = { "variant" => variant, "seed" => seed, "step" => trial.state.fetch("step"), "checkpoint" => trial.last_checkpoint,
      "parameters" => trial.model.parameter_count, "device" => trial.device, "known_mastery" => history.any? { |cell| cell.fetch("mastery_checks") >= 2 },
      "training_full" => full, "development" => measured, "examples_seen" => trial.state.fetch("examples_seen"),
      "row_visits_sha256" => Digest::SHA256.hexdigest(JSON.generate(metadata.fetch("row_visits"))),
      "mean_original_row_visits" => metadata.fetch("row_visits").sum.fdiv(metadata.fetch("row_visits").size),
      "unvisited_original_rows" => metadata.fetch("row_visits").count(&:zero?), "input_tokens" => metadata.fetch("input_tokens"),
      "session_seconds" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - started,
      "scope" => "Observed controlled binary direction experiment; augmented surfaces are not counted as unique original examples. No acceptance or factual release." }
    write(result_path, value)
    puts "Completed v07 #{variant}/#{seed} #{steps}"
  end

  def direction?(candidate, reference, rules)
    names = %w[canonical seen_combinations novel_actor]
    groups = lambda do |value|
      names.flat_map { |name| value.fetch("development").fetch(name).fetch("by_language").values.map { |cell| cell.fetch("actor_binding_all_correct") } }.min
    end
    improved = groups.call(candidate) + 1e-12 >= groups.call(reference) + rules.fetch("worst_actor_gain")
    improved && names.all? do |name|
      %w[en-US zh-CN].all? do |language|
        candidate.dig("development", name, "by_language", language, "balanced") + rules.fetch("maximum_balanced_drop") + 1e-12 >= reference.dig("development", name, "by_language", language, "balanced")
      end
    end
  end

  def select(root)
    protocol = verify(root)
    path = File.join(root, "selection.json")
    raise "Preserve v07 selection" if File.exist?(path)
    paths = VARIANTS.to_h { |variant| [variant, File.join(root, "arms", "#{variant}-1337/result-6000.json")] }
    results = paths.transform_values { |file| JSON.parse(File.read(file)) }
    baseline = results.fetch("ce")
    improved = results.except("ce").select { |_, value| direction?(value, baseline, protocol.fetch("direction")) }
    winner = improved.max_by do |_, value|
      required = value.fetch("development").slice("canonical", "seen_combinations", "novel_actor").values
      [required.flat_map { |metrics| metrics.fetch("by_language").values.map { |cell| cell.fetch("actor_binding_all_correct") } }.min,
        required.map { |metrics| metrics.fetch("worst_language_balanced") }.min, -required.sum { |metrics| metrics.fetch("nll") }]
    end
    history = JSON.parse(File.read(File.join(root, "arms/ce-1337/development.json")))
    earlier = history.find { |cell| cell.fetch("step") == 5000 }.fetch("training_probe")
    final = history.find { |cell| cell.fetch("step") == 6000 }.fetch("training_probe")
    extend_ce = final.fetch("worst_language_balanced") + 1e-12 >= earlier.fetch("worst_language_balanced") + 0.01 || DecisionScaling.minimum_groups(final) + 1e-12 >= DecisionScaling.minimum_groups(earlier) + 0.03
    value = { "improving" => !winner.nil?, "winner" => winner&.first, "improving_variants" => improved.keys, "extend_ce" => extend_ce,
      "results_sha256" => paths.transform_values { |file| sha(file) }, "scope" => "Directional selection at matched original-row visits; not release eligibility. CE extension stays separate from the fixed factorial." }
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts JSON.pretty_generate(value)
  end

  def decision(root)
    path = File.join(root, "selection.json")
    raise "v07 selection changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    value = JSON.parse(File.read(path))
    raise "v07 selection arms changed" unless value.fetch("results_sha256").keys.sort == VARIANTS.sort
    value.fetch("results_sha256").each do |variant, digest|
      file = File.join(root, "arms", "#{variant}-1337", "result-6000.json")
      raise "v07 selected result changed" unless sha(file) == digest
    end
    value
  end

  def report(root)
    protocol = verify(root)
    traces = ledger(root)
    traces.each_key do |name|
      File.foreach(File.join(root, name)) do |line|
        row = JSON.parse(line)
        raise "Invalid v07 optimizer trace" unless row.fetch("device") == "cuda" && row.values.grep(Numeric).all?(&:finite?)
      end
    end
    results = Dir[File.join(root, "arms/*/result-*.json")].sort.to_h { |file| [file.delete_prefix(root + "/"), JSON.parse(File.read(file)).merge("result_sha256" => sha(file))] }
    primary = results.select { |name, _| name.end_with?("1337/result-6000.json") }
    raise "v07 original row sampling differs across interventions" if primary.values.map { |value| value.fetch("row_visits_sha256") }.uniq.size > 1
    value = { "protocol_sha256" => sha(File.join(root, "protocol.json")), "results" => results, "update_ledger" => traces,
      "actual_optimizer_updates" => traces.values.sum, "update_ceiling" => UPDATE_CEILING,
      "historical_equal_mean_exposure_reference" => JSON.parse(File.read(File.join(ROOT_V06, "arms/base-32-1337/screen.json"))).except("checkpoint"),
      "acceptance_opened" => false, "scope" => "All measurements are observed development or training diagnostics; no factual release or Jev parity claim." }
    value["selection"] = decision(root) if File.exist?(File.join(root, "selection.json"))
    budget!(root, 0)
    write(File.join(root, "report.json"), value)
    puts "v07 report: #{results.size} endpoints, #{traces.values.sum} actual CUDA updates"
    value
  end
end
