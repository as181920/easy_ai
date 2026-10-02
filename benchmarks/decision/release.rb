#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require_relative "../../lib/easy_ai"
require_relative "semantic_coverage_evaluation"
require "optparse"
require "rbconfig"

# Product delivery runner: one supported profile, a closed acceptance panel, no business integration.
module DecisionRelease
  ROOT = File.expand_path("../..", __dir__)
  PILOT = File.join(ROOT, "runs/decision/natural-v1-pilot")
  ARCHIVE = File.join(ROOT, "data/decision/downloads/massive-1.1.tar.gz")
  ARCHIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577".freeze
  LANGUAGES = %w[en-US zh-CN].freeze
  EVALUATION_UNIT = "One example per language/material-ID component, chosen by fixed ID hash without targets.".freeze
  STEPS = 2500
  module_function

  def write(path, value)
    EasyAI::Decision::Checkpoint.atomic_json(path, value)
  end

  def rows(path)
    EasyAI::Decision::Data::Dataset.new(path).to_a
  end

  def history
    paths = (Dir.glob(File.join(ROOT, "data/decision/**/*.jsonl")) + Dir.glob(File.join(ROOT, "runs/decision/**/data/*.jsonl")))
      .reject { |path| path.include?("/downloads/") || path.include?("/release-v01/") }.sort
    material, training, hashes, seen = Set.new, Set.new, {}, {}
    paths.each do |path|
      sha = Digest::SHA256.file(path).hexdigest
      hashes[path] = sha
      kind = %w[train.jsonl corpus.jsonl broad.jsonl mixed.jsonl control.jsonl sanity.jsonl].include?(File.basename(path))
      if seen.key?(sha)
        material.merge(seen.fetch(sha))
        training.merge(seen.fetch(sha)) if kind
        next
      end
      values = Set.new
      File.foreach(path) do |line|
        next if line.strip.empty?
        row = JSON.parse(line)
        text = row["state"] || row["text"]
        values << EasyAI::Decision::Data::NaturalCorpus.material(text) if text.is_a?(String)
      end
      seen[sha] = values
      material.merge(values)
      training.merge(values) if kind
    end
    %w[validation calibration].each do |name|
      rows(File.join(PILOT, "data/#{name}.jsonl")).each { |row| training << EasyAI::Decision::Data::NaturalCorpus.material(row.state) }
    end
    [material, training, hashes]
  end

  def panel(raw, count)
    groups = raw.group_by { |row| row.fetch("group_id") }
    # Keep both languages whenever present; choose groups independently of labels.
    selected = groups.keys.sort_by { |group| Digest::SHA256.hexdigest("release-panel-v01:#{group}") }.first(count)
    selected.flat_map { |group| groups.fetch(group) }
  end

  def prepare(root)
    raise ArgumentError, "Release run exists" if File.exist?(root)
    raise "MASSIVE archive changed" unless Digest::SHA256.file(ARCHIVE).hexdigest == ARCHIVE_SHA256
    loaded = EasyAI::Decision::Checkpoint.load(File.join(PILOT, "broad-sinusoidal/selected"))
    config = loaded.fetch(:model).config.with(training: { steps: STEPS, learning_rate: 0.0003,
      warmup_steps: 100, eval_every: 200, checkpoint_every: 500, early_stopping_patience: 6,
      choice_microbatch: 4, gradient_accumulation: 8, balance_sources: false, balance_labels: true, track_coverage: true })
    train = rows(File.join(PILOT, "data/broad.jsonl")).select { |row| row.source == "MASSIVE-Scenario" }
    eligible = train.map { |row| EasyAI::Decision::Data::NaturalCorpus.material(row.state) }.to_set
    observed, trained_material, historical_files = history
    raw = EasyAI::Decision::Data::NaturalAdapter.massive(ARCHIVE, partitions: %w[train dev test]).to_a
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: config)
    corpus = EasyAI::Decision::Data::RoutingCorpus.new(raw, historical_material: observed, train_material: eligible,
      training_material: trained_material, collator: collator)
    splits = corpus.splits
    splits["validation"] = panel(splits.fetch("validation"), 400)
    splits["calibration"] = panel(splits.fetch("calibration"), 600)
    splits["test"] = panel(splits.fetch("test"), 800)
    splits.each do |name, data|
      counts = data.group_by { |row| row.fetch("language") }.transform_values(&:size)
      raise "Insufficient #{name} data: #{counts}" unless LANGUAGES.all? { |language| counts.fetch(language, 0) >= (name == "train" ? 1000 : 300) }
    end
    FileUtils.mkdir_p(File.join(root, "data"))
    splits.each do |name, data|
      File.open(File.join(root, "data/#{name}.jsonl"), "w") { |file| data.each { |row| file.puts(JSON.generate(row)) } }
    end
    datasets = splits.keys.map { |name| EasyAI::Decision::Data::Dataset.new(File.join(root, "data/#{name}.jsonl")) }
    EasyAI::Decision::Data::Dataset.assert_disjoint!(*datasets)
    material_sets = datasets.map { |data| data.map { |row| EasyAI::Decision::Data::NaturalCorpus.material(row.state) }.to_set }
    material_sets.combination(2).each { |left, right| raise "Release material leakage" unless (left & right).empty? }
    model = EasyAI::Decision::ChoiceModel.new(config)
    model.load_state_dict(loaded.fetch(:model).state_dict)
    initial = EasyAI::Decision::Checkpoint.save(File.join(root, "initial"), model: model, tokenizer: loaded.fetch(:tokenizer))
    protocol = { "version" => "0.1", "profile" => "bilingual-request-domains", "seed" => 1337,
      "steps" => STEPS, "config" => config.to_h, "initial" => initial,
      "initial_sha256" => Digest::SHA256.file(File.join(initial, "weights.pt")).hexdigest,
      "source" => { "archive" => ARCHIVE, "sha256" => ARCHIVE_SHA256, "license" => "CC-BY-4.0" },
      "parent_sha256" => loaded.fetch(:weights_fingerprint), "historical_files_sha256" => historical_files,
      "requirements" => EasyAI::Decision::ReleasePolicy::REQUIREMENTS, "languages" => LANGUAGES,
      "files_sha256" => Dir.glob(File.join(root, "data/*")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
      "counts" => splits.transform_values { |data| data.group_by { |row| row.fetch("language") }.transform_values(&:size) },
      "exclusions" => corpus.exclusions,
      "scope" => "18 full-candidate MASSIVE request domains; own pretrained-by-us parent, no external weights. Official train only; official dev for validation/calibration and unused official test for acceptance. Parallel IDs and material components never cross splits. Dev excludes all historical training and the initializer selection/calibration material; test excludes ALL historical prepared material. Historical dev diagnostics are not claimed fresh. Supported lengths only. Thresholds calibrated before acceptance; no business integration or universal semantic claim." }
    write(File.join(root, "protocol.json"), protocol)
    write(File.join(root, "evaluation-unit.json"), { "evaluation_unit" => EVALUATION_UNIT, "frozen_before_calibration_and_acceptance" => true })
    puts JSON.pretty_generate(protocol.slice("counts", "exclusions", "requirements"))
  end

  def verify(root)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    unit = JSON.parse(File.read(File.join(root, "evaluation-unit.json")))
    raise "Evaluation unit changed" unless unit.fetch("evaluation_unit") == EVALUATION_UNIT
    raise "Acceptance requirements changed" unless protocol.fetch("requirements") == EasyAI::Decision::ReleasePolicy::REQUIREMENTS
    protocol.fetch("files_sha256").each do |name, sha|
      raise "Prepared release data changed: #{name}" unless Digest::SHA256.file(File.join(root, "data", name)).hexdigest == sha
    end
    protocol
  end

  def train(root)
    protocol = verify(root)
    raise "Release already trained" if File.exist?(File.join(root, "choice"))
    loaded = EasyAI::Decision::Checkpoint.load(protocol.fetch("initial"))
    raise "Initial weights changed" unless loaded.fetch(:weights_fingerprint) == protocol.fetch("initial_sha256")
    ENV["EASY_AI_LOG_PATH"] = File.join(root, "train.log")
    EasyAI::Logger.reset!
    trainer = EasyAI::Decision::CandidateTrainer.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer),
      dataset: EasyAI::Decision::Data::Dataset.new(File.join(root, "data/train.jsonl")), output: File.join(root, "choice"),
      validation: EasyAI::Decision::Data::Dataset.new(File.join(root, "data/validation.jsonl")))
    progress = EasyAI::Decision::Progress.new(task: "decision-v0.1", total: protocol.fetch("steps"))
    trainer.train { |state, loss| progress.update(state, loss, device: trainer.device) }
    selected = EasyAI::Decision::Checkpoint.resolve(File.join(root, "choice/best"))
    FileUtils.cp_r(selected, File.join(root, "selected"))
    write(File.join(root, "summary.json"), trainer.state.slice("step", "best_step", "best_validation_loss", "examples_seen", "stop_reason").merge("device" => trainer.device))
    puts JSON.pretty_generate(EasyAI::Decision::TrainingReport.new(root).write)
  end

  def baseline(root)
    verify(root)
    raise "Baseline already evaluated" if File.exist?(File.join(root, "baseline-validation.json"))
    loaded = EasyAI::Decision::Checkpoint.load(File.join(root, "initial"))
    data, logits = collect(root, "validation", loaded, "cpu")
    calibrator = EasyAI::Decision::Calibrator.new
    metrics = LANGUAGES.to_h do |language|
      indexes = data.each_index.select { |index| data[index].language == language }
      [language, EasyAI::Decision::ReleaseMetrics.measure(indexes.map { |index| calibrator.probabilities(logits[index]) },
        indexes.map { |index| data[index].target_index }, threshold: 1.0)]
    end
    write(File.join(root, "baseline-validation.json"), { "by_language" => metrics,
      "scope" => "Own initializer on the release validation panel, not acceptance.", "weights_sha256" => loaded.fetch(:weights_fingerprint) })
    puts JSON.pretty_generate(metrics)
  end

  def collect(root, name, loaded, device)
    data = EasyAI::Decision::Data::RoutingCorpus.independent_rows(rows(File.join(root, "data/#{name}.jsonl")))
    evaluator = SemanticCoverageEvaluation.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), device: device, batch_size: 4)
    logits = evaluator.collect(data)
    EasyAI::Runtime::DevicePolicy.new(requested: device, budget_mib: 4096).check_budget!(device)
    [data, logits]
  rescue Torch::Error, EasyAI::Runtime::DevicePolicy::MemoryBudgetExceeded => error
    policy = EasyAI::Runtime::DevicePolicy.new(requested: device, budget_mib: 4096)
    raise unless device == "cuda" && policy.recoverable?(error)
    EasyAI.logger.warn("Release evaluation falling back to CPU: #{error.message.lines.first}")
    loaded.fetch(:model).to("cpu")
    GC.start
    collect(root, name, loaded, "cpu")
  end

  def calibrate(root)
    protocol = verify(root)
    raise "Calibration already frozen" if File.exist?(File.join(root, "policy.json"))
    loaded = EasyAI::Decision::Checkpoint.load(File.join(root, "selected"))
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    data, logits = collect(root, "calibration", loaded, device)
    calibrator = EasyAI::Decision::Calibrator.new.fit(logits, data.map(&:target_index))
    policies = LANGUAGES.to_h do |language|
      indexes = data.each_index.select { |index| data[index].language == language }
      guard = protocol.fetch("calibration_guard", EasyAI::Decision::ReleasePolicy::REQUIREMENTS)
      [language, EasyAI::Decision::ReleasePolicy.fit(indexes.map { |index| calibrator.probabilities(logits[index]) },
        indexes.map { |index| data[index].target_index }, minimum_accuracy: guard.fetch("accepted_accuracy"),
        minimum_lower_bound: guard.fetch("accepted_accuracy_lower_95"))]
    end
    calibration = { "temperature" => calibrator.temperature, "groups" => data.map(&:group_id).uniq,
      "dataset_sha256" => Digest::SHA256.file(File.join(root, "data/calibration.jsonl")).hexdigest }
    path = EasyAI::Decision::Checkpoint.save(File.join(root, "calibrated"), model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer),
      training_state: loaded.fetch(:metadata).fetch("training"), calibration: calibration)
    write(File.join(root, "policy.json"), { "checkpoint" => path, "weights_sha256" => Digest::SHA256.file(File.join(path, "weights.pt")).hexdigest,
      "temperature" => calibrator.temperature, "languages" => policies,
      "evaluation_unit" => EVALUATION_UNIT,
      "evaluation_protocol_sha256" => Digest::SHA256.file(File.join(root, "evaluation-unit.json")).hexdigest,
      "device" => loaded.fetch(:model).parameters.first.device.type.to_s })
    puts JSON.pretty_generate(policies)
  end

  def acceptance(root)
    protocol = verify(root)
    raise "Acceptance already opened; use a new prospectively frozen panel for retuning" if File.exist?(File.join(root, "acceptance-opened.json"))
    policy = JSON.parse(File.read(File.join(root, "policy.json")))
    raise "Evaluation protocol changed" unless policy.fetch("evaluation_protocol_sha256") == Digest::SHA256.file(File.join(root, "evaluation-unit.json")).hexdigest
    write(File.join(root, "acceptance-opened.json"), { "time" => Time.now.utc.iso8601, "policy_sha256" => Digest::SHA256.file(File.join(root, "policy.json")).hexdigest })
    loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
    raise "Calibrated weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    data, logits = collect(root, "test", loaded, device)
    calibrator = EasyAI::Decision::Calibrator.new(temperature: policy.fetch("temperature"))
    probabilities = logits.map { |values| calibrator.probabilities(values) }
    metrics = LANGUAGES.to_h do |language|
      indexes = data.each_index.select { |index| data[index].language == language }
      threshold = policy.fetch("languages").fetch(language).fetch("threshold")
      measured = EasyAI::Decision::ReleaseMetrics.measure(indexes.map { |index| probabilities[index] }, indexes.map { |index| data[index].target_index },
        threshold: threshold || 1.0)
      failures = EasyAI::Decision::ReleasePolicy.failures(measured, expected_labels: (0...18).to_a)
      failures << "No calibration policy meets automation requirements" unless threshold
      [language, measured.merge("failures" => failures)]
    end
    File.open(File.join(root, "acceptance-predictions.jsonl"), "w") do |file|
      data.each_with_index do |row, index|
        file.puts(JSON.generate(row.to_h.merge("logits" => logits[index], "probabilities" => probabilities[index])))
      end
    end
    report = { "passed" => metrics.values.all? { |value| value.fetch("failures").empty? }, "by_language" => metrics,
      "device" => loaded.fetch(:model).parameters.first.device.type.to_s, "policy_sha256" => Digest::SHA256.file(File.join(root, "policy.json")).hexdigest,
      "protocol_sha256" => Digest::SHA256.file(File.join(root, "protocol.json")).hexdigest,
      "scope" => protocol.fetch("scope"), "evaluation_unit" => policy.fetch("evaluation_unit") }
    write(File.join(root, "acceptance.json"), report)
    puts JSON.pretty_generate(report)
  end

  def preview(root)
    package(root, preview: true)
  end

  def package(root, preview: false)
    verify(root)
    report = JSON.parse(File.read(File.join(root, "acceptance.json")))
    raise "Model failed v0.1 acceptance; retain it as a diagnostic" unless preview || report.fetch("passed")
    policy = JSON.parse(File.read(File.join(root, "policy.json")))
    runtime = runtime_check(root, policy.fetch("checkpoint"))
    write(File.join(root, "runtime.json"), runtime)
    raise "Runtime verification failed" unless runtime.fetch("passed")
    destination = File.join(ROOT, preview ? "runs/decision/v0.1-preview" : "runs/decision/v0.1")
    puts EasyAI::Decision::Release.publish(run: root, output: destination, preview: preview)
  end

  def runtime_check(root, checkpoint)
    samples = rows(File.join(root, "data/validation.jsonl")).group_by(&:language).values.map(&:first)
    cpu = EasyAI::Decision::Predictor.load(checkpoint, device: "cpu", candidate_chunk_size: 18)
    cuda = EasyAI::Decision::Predictor.load(checkpoint, device: "cuda", candidate_chunk_size: 18)
    comparisons = samples.map do |row|
      input = { state: row.state, question: row.question, options: row.options }
      cpu_values = cpu.probabilities(**input).fetch("probabilities")
      gpu_values = cuda.probabilities(**input).fetch("probabilities")
      reversed = cuda.probabilities(**input.merge(options: row.options.reverse)).fetch("probabilities")
      { "language" => row.language, "cpu_cuda_max_delta" => cpu_values.keys.map { |id| (cpu_values.fetch(id) - gpu_values.fetch(id)).abs }.max,
        "permutation_max_delta" => gpu_values.keys.map { |id| (gpu_values.fetch(id) - reversed.fetch(id)).abs }.max }
    end
    memory = EasyAI::Runtime::DevicePolicy.new(requested: "cuda", budget_mib: 4096)
    before = memory.process_memory_mib
    latency = { "cpu" => cpu, "cuda" => cuda }.transform_values do |predictor|
      elapsed = samples.cycle.take(10).map do |row|
        start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        predictor.probabilities(state: row.state, question: row.question, options: row.options)
        (Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) * 1000
      end
      { "mean_ms" => elapsed.sum / elapsed.size, "max_ms" => elapsed.max, "calls" => elapsed.size }
    end
    after = memory.check_budget!(cuda.device)
    loaded = EasyAI::Decision::Checkpoint.load(checkpoint)
    { "passed" => cuda.device == "cuda" && comparisons.all? do |result|
        result.fetch("cpu_cuda_max_delta") <= 1e-4 && result.fetch("permutation_max_delta") <= 1e-4
      end,
      "comparisons" => comparisons, "latency" => latency, "cuda_memory_before_mib" => before, "cuda_memory_after_mib" => after,
      "weights_sha256" => loaded.fetch(:weights_fingerprint),
      "candidate_chunk_size" => 18,
      "scope" => "Two validation examples, warm cached 18-candidate API calls; latency is a local smoke measurement, not a production distribution benchmark." }
  end

  def all(root)
    %w[prepare baseline train calibrate acceptance package].each do |phase|
      raise "Release #{phase} failed" unless system(RbConfig.ruby, __FILE__, "--phase", phase, "--output", root)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/release-v01-routing" }
  OptionParser.new do |parser|
    %i[phase output].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  end.parse!
  root = File.expand_path(options[:output])
  raise ArgumentError, "Unknown release phase" unless %w[all prepare baseline train calibrate acceptance package preview].include?(options[:phase])
  DecisionRelease.public_send(options[:phase], root)
end
