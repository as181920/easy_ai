module DecisionScaling
  module_function

  def prepare_memory(root)
    path = File.join(root, "preflight/protocol.json")
    raise "Preflight already frozen" if File.exist?(path)
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    files = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort
    files.each do |file|
      destination = File.join(root, "preflight/source-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
    end
    value = { "profiles" => PROFILES, "parent_sha256" => loaded.fetch(:weights_fingerprint), "tokenizer_fingerprint" => loaded.fetch(:tokenizer).fingerprint,
      "source_sha256" => files.to_h { |file| [file, sha(file)] }, "plan_sha256" => sha("docs/decision/v06-plan.md"),
      "memory_budget_mib" => 4096, "effective_batch" => 32, "candidate_counts" => [2, 3, 18], "steps_per_count" => 3,
      "scope" => "FP32 CUDA forward/backward/AdamW at full input limits. Artificial random token inputs test feasibility, not semantics or learning quality." }
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts "Frozen CUDA capacity preflight"
  end

  def memory_batch(config, count, microbatch, lengths: nil)
    state = lengths ? lengths.fetch("state") : config[:input]["state_max_tokens"]
    option = lengths ? lengths.fetch("option") : config[:input]["question_option_max_tokens"]
    raise ArgumentError, "Probe exceeds model input limits" unless state <= config[:input]["state_max_tokens"] && option <= config[:input]["question_option_max_tokens"]
    vocab = config[:model]["vocab_size"]
    { state_ids: Torch.randint(vocab - 4, [microbatch, state], dtype: :int64, device: "cuda") + 4,
      state_mask: Torch.ones([microbatch, state], dtype: :bool, device: "cuda"),
      option_ids: Torch.randint(vocab - 4, [microbatch, count, option], dtype: :int64, device: "cuda") + 4,
      option_mask: Torch.ones([microbatch, count, option], dtype: :bool, device: "cuda"),
      candidate_mask: Torch.ones([microbatch, count], dtype: :bool, device: "cuda"),
      targets: Torch.tensor(Array.new(microbatch) { |index| index % count }, dtype: :int64, device: "cuda") }
  end

  def memory(root, profile, microbatch, controlled: false)
    directory = File.join(root, "preflight", "#{profile}-#{controlled ? 'controlled-' : ''}micro#{microbatch}")
    raise "Preserve completed memory trial" if File.exist?(File.join(directory, "result.json"))
    protocol_path = File.join(root, "preflight/protocol.json")
    raise "Preflight protocol changed" unless sha(protocol_path) == JSON.parse(File.read(protocol_path + ".sha256.json")).fetch("sha256")
    protocol = JSON.parse(File.read(protocol_path))
    lengths = nil
    if controlled
      audit = JSON.parse(File.read(File.join(root, "data-audit.json")))
      lengths = audit.fetch("token_limits_observed").transform_values { |value| ((value + 15) / 16) * 16 }
      protocol = protocol.merge("candidate_counts" => [2], "scope" => "Controlled-only binary feasibility at rounded-up audited corpus lengths; broader input/candidate limits require separate full-shape probes.")
    end
    raise "Parent changed" unless sha(File.join(PARENT, "weights.pt")) == protocol.fetch("parent_sha256")
    raise "CUDA required for memory trial" unless Torch::CUDA.available?
    FileUtils.mkdir_p(directory)
    model, _, transfer = initialize_model(profile, microbatch: microbatch)
    transfer["probe_lengths"] = lengths || model.config[:input].slice("state_max_tokens", "question_option_max_tokens")
    transfer["scenario"] = controlled ? "controlled" : "full"
    transfer["plan_sha256"] = sha("docs/decision/v06-plan.md")
    FileUtils.cp("docs/decision/v06-plan.md", File.join(directory, "plan-frozen.md"))
    transfer["runtime_source_sha256"] = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort.to_h do |file|
      destination = File.join(directory, "source-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
      [file, sha(destination)]
    end
    write(File.join(directory, "initialization.json"), transfer)
    model.to("cuda").train
    optimizer = EasyAI::Optim::AdamW.new(model.named_parameters, learning_rate: 0.0003, weight_decay: 0.01)
    policy = EasyAI::Runtime::DevicePolicy.new(requested: :cuda, budget_mib: 4096)
    values = []
    begin
      protocol.fetch("candidate_counts").each do |count|
        protocol.fetch("steps_per_count").times do |step|
          started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          optimizer.zero_grad
          loss_value, samples = 0.0, []
          (32 / microbatch).times do |micro|
            Torch.manual_seed(1337 + step * 1009 + micro)
            batch = memory_batch(model.config, count, microbatch, lengths: lengths)
            loss = Torch::NN::Functional.cross_entropy(model.call(batch), batch.fetch(:targets))
            raise "Nonfinite probe loss" unless loss.item.finite?
            samples << policy.check_budget!("cuda") if micro.zero?
            (loss / (32 / microbatch)).backward
            samples << policy.check_budget!("cuda") if micro.zero?
            loss_value += loss.item / (32 / microbatch)
            batch, loss = nil, nil
            GC.start
          end
          gradient = optimizer.clip_grad_norm!(1.0)
          optimizer.step
          sample = { "candidates" => count, "step" => step + 1, "loss" => loss_value, "gradient_norm_preclip" => gradient, "device" => "cuda",
            "seconds" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - started, "sampled_process_mib" => samples }
          File.open(File.join(directory, "training.jsonl"), "a") { |file| file.puts(JSON.generate(sample)) }
          sample["sampled_process_mib"] << policy.check_budget!("cuda")
          values << sample
          puts "Memory #{profile}/micro#{microbatch} candidates=#{count} step=#{step + 1} MiB=#{samples.compact.max}"
          $stdout.flush
        end
      end
      write(File.join(directory, "result.json"), { "passed" => true, "parameters" => model.parameter_count, "microbatch" => microbatch,
        "measurements" => values, "scope" => protocol.fetch("scope") + " Memory is sampled, not continuous peak; no CPU fallback." })
    rescue Torch::Error, EasyAI::Runtime::DevicePolicy::MemoryBudgetExceeded => error
      trace = File.join(directory, "training.jsonl")
      write(File.join(directory, "result.json"), { "passed" => false, "error" => error.message.lines.first, "completed_updates" => File.exist?(trace) ? File.foreach(trace).count : 0,
        "parameters" => model.parameter_count, "microbatch" => microbatch, "scope" => "Resource failure, not a negative semantic learning result" })
      puts "Memory trial failed: #{error.message.lines.first}"
    end
  end
end
