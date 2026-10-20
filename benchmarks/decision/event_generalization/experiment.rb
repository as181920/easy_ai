module DecisionEventGeneralization
  module_function

  def evaluate_views(root, model, tokenizer, names, identity, device: "cuda")
    raise "Lockbox remains closed" if names.any? { |name| name.start_with?("lockbox-") } && !File.exist?(File.join(root, "lockbox-opened.json"))
    directory = File.join(root, "predictions", identity)
    FileUtils.mkdir_p(directory)
    names.to_h do |name|
      path = File.join(directory, "#{name}.json")
      if File.exist?(path)
        value = sealed(path)
        raise "Prediction rows changed" unless sha(File.join(directory, "#{name}.jsonl")) == value.fetch("predictions_sha256")
      else
        rows = raw(root, name)
        _, logits = DecisionRobust.collect(model, tokenizer, rows, device, gc_interval: 8)
        raise "v08 inference device fallback" unless model.parameters.all? { |parameter| parameter.device.to_s.start_with?(device) }
        metrics = DecisionJudgment.metrics(rows, logits)
        predictions = rows.each_index.map { |index| rows[index].fetch("options")[logits[index].each_index.max_by { |option| logits[index][option] }].fetch("id") }
        independent = %w[actor event].to_h { |axis| [axis, independent_pairs(rows, predictions, axis: axis)] }
        independent.each do |axis, languages|
          languages.each do |language, cell|
            raise "Independent #{axis} pair mismatch" unless cell.fetch("groups") == metrics.dig("by_language", language, "#{axis}_binding_groups") && cell.fetch("all_correct") == metrics.dig("by_language", language, "#{axis}_binding_all_correct")
          end
        end
        prediction_path = File.join(directory, "#{name}.jsonl")
        DecisionRobust.write_rows(prediction_path, rows.each_index.map { |index| rows[index].merge("prediction" => predictions[index], "logits" => logits[index]) })
        value = seal(path, { "metrics" => metrics, "independent_pairs" => independent, "predictions_sha256" => sha(prediction_path), "device" => device })
        puts "Measured #{identity}/#{name}: #{metrics.fetch('worst_language_balanced').round(4)}"
        $stdout.flush
      end
      [name, value.fetch("metrics")]
    end
  end

  def preflight(root, variant)
    verify(root)
    raise "Unregistered feasibility variant" unless VARIANTS.include?(variant.to_s)
    path = File.join(root, "preflight", variant.to_s, "result.json")
    return sealed(path) if File.exist?(path)
    budget!(root, 3)
    trial = trainer(root, variant, 1337, File.join(root, "preflight", variant.to_s, "choice"))
    raise "Feasibility requires actual CUDA" unless trial.device == "cuda"
    collator = trial.instance_variable_get(:@collator)
    original = collator.method(:call)
    collator.define_singleton_method(:call) do |examples, **options|
      batch = original.call(examples, **options)
      size = examples.size
      batch[:state_ids] = Torch.full([size, 256], 5, dtype: :int64, device: options.fetch(:device))
      batch[:state_mask] = Torch.ones([size, 256], dtype: :bool, device: options.fetch(:device))
      batch[:option_ids] = Torch.full([size, 2, 128], 5, dtype: :int64, device: options.fetch(:device))
      batch[:option_mask] = Torch.ones([size, 2, 128], dtype: :bool, device: options.fetch(:device))
      batch[:answer_mask] = batch[:option_mask]
      @input_token_counts = Array.new(size, 512)
      batch
    end
    memory = []
    trial.train(steps: 3) do
      raise "Feasibility silently fell back" unless trial.device == "cuda"
      memory << EasyAI::Runtime::DevicePolicy.new(requested: :cuda).process_memory_mib
    end
    seal(path, { "passed" => true, "state_tokens" => 256, "option_tokens" => 128, "microbatch" => 16, "accumulation" => 2, "sampled_process_mib" => memory, "steps" => 3, "device" => trial.device })
  end

  def baseline(root, seed)
    verify(root)
    path = File.join(root, "baseline-#{seed}.json")
    return sealed(path) if File.exist?(path)
    loaded = EasyAI::Decision::Checkpoint.load(parent(seed))
    values = evaluate_views(root, loaded.fetch(:model), loaded.fetch(:tokenizer), Catalog::VIEWS.map { |view| "development-#{view}" }, "baseline-#{seed}")
    seal(path, { "seed" => seed, "checkpoint" => parent(seed), "weights_sha256" => loaded.fetch(:weights_fingerprint), "development" => values.transform_keys { |name| name.delete_prefix("development-") }, "scope" => "Fresh development before v08 training; not the unopened lockbox." })
  end

  def train(root, variant, seed)
    variant = variant.to_s
    raise "Unregistered arm" unless VARIANTS.include?(variant) && [1337, 2027].include?(seed)
    FileUtils.mkdir_p(File.join(root, "locks"))
    File.open(File.join(root, "locks", "#{variant}-#{seed}.lock"), "w") do |lock|
      lock.flock(File::LOCK_EX)
      verify(root)
      path = File.join(root, "arms", "#{variant}-#{seed}", "result.json")
      if File.exist?(path)
        completed = sealed(path)
        raise "Completed arm identity mismatch" unless completed.values_at("variant", "seed", "step") == [variant, seed, 6000]
        return completed
      end
      if seed == 2027
        decision = selection(root)
        raise "No selected recipe to confirm" unless decision.fetch("improving") && ["reference", decision.fetch("winner")].include?(variant)
      end
      raise "No optimization after lockbox opening" if File.exist?(File.join(root, "lockbox-opened.json"))
      VARIANTS.each { |arm| raise "Failed feasibility" unless sealed(File.join(root, "preflight", arm, "result.json")).fetch("passed") }
      [1337, 2027].each { |number| sealed(File.join(root, "baseline-#{number}.json")) }
      directory = File.dirname(path)
      FileUtils.mkdir_p(directory)
      ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
      EasyAI::Logger.reset!
      output = File.join(directory, "choice")
      trial = trainer(root, variant, seed, output, resume: File.exist?(File.join(output, "latest.json")))
      raise "Wrong training seed" unless trial.model.config[:training]["seed"] == seed
      budget!(root, 6000 - trial.state.fetch("step"))
      history_path = File.join(directory, "development.json")
      history = File.exist?(history_path) ? JSON.parse(File.read(history_path)).select { |row| row.fetch("step") <= trial.state.fetch("step") } : []
      trial.train do |state, loss|
        raise "v08 training device/layout changed" unless trial.device == "cuda" && trial.instance_variable_get(:@microbatch) == 16 && trial.instance_variable_get(:@accumulation) == 2
        if (state.fetch("step") % 100).zero?
          puts "v08 #{variant}/#{seed} step=#{state.fetch('step')} loss=#{loss.round(5)}"
          $stdout.flush
        end
        next unless (state.fetch("step") % 500).zero?
        pointers = trial.model.named_parameters.transform_values(&:object_id)
        probes = %w[probe probe-augmented].to_h { |name| [name, DecisionScaling.measure(trial.model, trial.tokenizer, raw(root, name), "cuda")] }
        curves = {}
        if (state.fetch("step") % 1000).zero?
          %w[combinations events expressions].each do |view|
            rows = raw(root, "development-#{view}").select { |row| row.fetch("group_id").split(":").last.to_i < 4 }
            curves[view] = DecisionScaling.measure(trial.model, trial.tokenizer, rows, "cuda")
          end
        end
        raise "Evaluation replaced optimizer parameters" unless pointers == trial.model.named_parameters.transform_values(&:object_id)
        history << { "step" => state.fetch("step"), "probes" => probes, "development_subset" => curves, "loss" => loss,
          "sampled_process_mib" => EasyAI::Runtime::DevicePolicy.new(requested: :cuda).process_memory_mib, "device" => trial.device }
        write(history_path, history)
        puts "v08 #{variant}/#{seed}: original=#{probes.fetch('probe').fetch('worst_language_balanced').round(4)} augmented=#{probes.fetch('probe-augmented').fetch('worst_language_balanced').round(4)}"
        $stdout.flush
      end
      names = Catalog::VIEWS.map { |view| "development-#{view}" } + %w[canonical seen_combinations novel_actor].map { |name| "regression-#{name}" }
      measured = evaluate_views(root, trial.model, trial.tokenizer, names, "#{variant}-#{seed}")
      full = DecisionScaling.measure(trial.model, trial.tokenizer, raw(root, "train"), "cuda")
      coverage = trial.state.fetch("coverage")
      rendering = trial.state.fetch("rendering_coverage")
      seal(path, { "variant" => variant, "seed" => seed, "step" => trial.state.fetch("step"), "checkpoint" => trial.last_checkpoint,
        "weights_sha256" => sha(File.join(trial.last_checkpoint, "weights.pt")), "parent_weights_sha256" => sha(File.join(parent(seed), "weights.pt")),
        "device" => trial.device, "parameters" => trial.model.parameter_count, "training_full" => full,
        "development" => measured.select { |name, _| name.start_with?("development-") }.transform_keys { |name| name.delete_prefix("development-") },
        "regression" => measured.select { |name, _| name.start_with?("regression-") }.transform_keys { |name| name.delete_prefix("regression-") },
        "row_visits_sha256" => Digest::SHA256.hexdigest(JSON.generate(coverage.fetch("row_visits"))), "examples_seen" => trial.state.fetch("examples_seen"),
        "mean_original_row_visits" => coverage.fetch("row_visits").sum.fdiv(coverage.fetch("row_visits").size), "unvisited_rows" => coverage.fetch("row_visits").count(&:zero?),
        "input_tokens" => coverage.fetch("input_tokens"), "rendering_coverage" => rendering.except("surface_hashes").merge("unique_state_surfaces" => rendering.fetch("surface_hashes").size),
        "source_sha256" => sha("benchmarks/decision/event_generalization/experiment.rb"), "scope" => "Fixed v08 endpoint, observed development and old regression. No lockbox predictions or production acceptance." })
    end
  end

  def select(root)
    verify(root)
    path = File.join(root, "selection.json")
    return selection(root) if File.exist?(path)
    results = VARIANTS.to_h { |variant| [variant, sealed(File.join(root, "arms", "#{variant}-1337", "result.json"))] }
    raise "v08 original sampling differs across arms" unless results.values.map { |value| value.fetch("row_visits_sha256") }.uniq.size == 1
    improved = results.except("reference").select { |_, value| direction?(value, results.fetch("reference")) }
    winner = improved.max_by { |_, value| rank(value) }
    seal(path, { "improving" => !winner.nil?, "winner" => winner&.first, "qualifying_variants" => improved.keys,
      "results_sha256" => VARIANTS.to_h { |variant| [variant, sha(File.join(root, "arms", "#{variant}-1337", "result.json"))] } })
  end

  def selection(root)
    value = sealed(File.join(root, "selection.json"))
    raise "Unregistered selection arms" unless value.fetch("results_sha256").keys.sort == VARIANTS.sort
    value.fetch("results_sha256").each { |variant, digest| raise "Selected endpoint changed" unless sha(File.join(root, "arms", "#{variant}-1337", "result.json")) == digest }
    value
  end

  def finalize_choices(root)
    verify(root)
    decision = selection(root)
    path = File.join(root, "final-choices.json")
    return sealed(path) if File.exist?(path)
    identities = { "baseline-1337" => parent(1337), "reference-1337" => sealed(File.join(root, "arms/reference-1337/result.json")).fetch("checkpoint") }
    confirmation = nil
    if decision.fetch("improving")
      winner = decision.fetch("winner")
      reference = sealed(File.join(root, "arms/reference-2027/result.json"))
      candidate = sealed(File.join(root, "arms", "#{winner}-2027", "result.json"))
      raise "Confirmation sampling differs" unless candidate.fetch("row_visits_sha256") == reference.fetch("row_visits_sha256")
      confirmation = direction?(candidate, reference)
      identities["baseline-2027"] = parent(2027)
      identities["reference-2027"] = reference.fetch("checkpoint")
      [1337, 2027].each { |seed| identities["#{winner}-#{seed}"] = sealed(File.join(root, "arms", "#{winner}-#{seed}", "result.json")).fetch("checkpoint") }
    end
    seal(path, { "selection_sha256" => sha(File.join(root, "selection.json")), "confirmation_passed" => confirmation,
      "checkpoints" => identities.transform_values { |checkpoint| { "path" => checkpoint, "weights_sha256" => sha(File.join(checkpoint, "weights.pt")) } },
      "scope" => "Fixed choices before lockbox opening, including unfavorable confirmation outcomes. No subsequent optimization." })
  end

  def lockbox(root)
    choices = finalize_choices(root)
    opened = File.join(root, "lockbox-opened.json")
    seal(opened, { "at" => Time.now.utc.iso8601, "choices_sha256" => sha(File.join(root, "final-choices.json")), "protocol_sha256" => sha(File.join(root, "protocol.json")) }) unless File.exist?(opened)
    raise "Lockbox choices changed" unless sealed(opened).fetch("choices_sha256") == sha(File.join(root, "final-choices.json"))
    choices.fetch("checkpoints").each do |identity, value|
      path = File.join(root, "lockbox", "#{identity}.json")
      next if File.exist?(path) && sealed(path)
      raise "Fixed lockbox weights changed" unless sha(File.join(value.fetch("path"), "weights.pt")) == value.fetch("weights_sha256")
      loaded = EasyAI::Decision::Checkpoint.load(value.fetch("path"))
      measured = evaluate_views(root, loaded.fetch(:model), loaded.fetch(:tokenizer), Catalog::VIEWS.map { |view| "lockbox-#{view}" }, identity)
      seal(path, { "weights_sha256" => value.fetch("weights_sha256"), "views" => measured.transform_keys { |name| name.delete_prefix("lockbox-") } })
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
    end
  end
end
