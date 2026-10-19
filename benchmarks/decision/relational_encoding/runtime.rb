module DecisionRelational
  module_function

  def runtime(root, arm, seed)
    verify(root)
    recipe_config(root, verify(root), arm)
    directory = File.join(root, "fit-#{arm}-#{seed}")
    result = JSON.parse(File.read(File.join(directory, "result.json")))
    loaded = EasyAI::Decision::Checkpoint.load(result.fetch("checkpoint"))
    model, tokenizer = loaded.values_at(:model, :tokenizer)
    raw = rows(File.join(root, "data/fit.jsonl")).group_by { |row| row.fetch("language") }.values.flat_map { |group| group.first(2) }
    examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    cpu = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: "cpu", batch_size: 2).collect(examples)
    requested = EasyAI::Runtime::DevicePolicy.new(requested: "auto").resolve
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: requested, batch_size: 2)
    values = evaluator.collect(examples)
    cold = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
    difference = cpu.zip(values).flat_map { |left, right| left.zip(right).map { |a, b| (a - b).abs } }.max
    raise "CPU/device logits diverged: #{difference}" unless difference < 1e-4
    evaluator.collect(examples)
    policy = EasyAI::Runtime::DevicePolicy.new(requested: requested, budget_mib: 4096)
    memory, latency = [], []
    8.times do
      before = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      evaluator.collect(examples)
      latency << Process.clock_gettime(Process::CLOCK_MONOTONIC) - before
      memory << policy.process_memory_mib if requested == "cuda"
      policy.check_budget!(requested)
    end
    output = { "device" => requested, "cpu_device_max_logit_difference" => difference,
      "cold_seconds_four_examples" => cold, "warm_seconds_four_examples" => latency,
      "process_mib" => memory, "checkpoint" => loaded.fetch(:path), "weights_sha256" => loaded.fetch(:weights_fingerprint),
      "scope" => "Numerical/device and repeated-inference diagnostic on four fitting records; no semantic acceptance or latency guarantee" }
    write(File.join(directory, "runtime.json"), output)
    puts JSON.pretty_generate(output)
  end
end
