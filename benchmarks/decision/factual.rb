#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require_relative "release"
require_relative "factual/preparation"
require_relative "factual/provenance"
require_relative "factual/evaluation"
require_relative "factual/report"
require "time"
$stdout.sync = true

module DecisionFactual
  ROOT = File.expand_path("../..", __dir__)
  INITIALIZERS = {
    "release" => File.join(ROOT, "runs/decision/v0.1-preview/checkpoint"),
    "broad" => File.join(ROOT, "runs/decision/natural-v1-pilot/broad-sinusoidal/selected")
  }.freeze
  ARMS = %w[ce margin].freeze
  STEPS = 2000
  module_function

  def all(root)
    %w[prepare audit baseline fit].each do |phase|
      raise "Factual #{phase} failed" unless system(RbConfig.ruby, __FILE__, "--phase", phase, "--output", root)
    end
    ARMS.each do |arm|
      raise "Factual training failed" unless system(RbConfig.ruby, __FILE__, "--phase", "train", "--output", root, "--arm", arm)
    end
    %w[confirm evaluate report].each do |phase|
      raise "Factual #{phase} failed" unless system(RbConfig.ruby, __FILE__, "--phase", phase, "--output", root)
    end
  end

  def config(seed)
    EasyAI::Decision::Config.new(JSON.parse(File.read(File.join(ROOT, "runs/decision/natural-v1-pilot/protocol.json"))).fetch("configs").fetch("sinusoidal"))
      .with(training: { seed: seed, steps: STEPS, eval_every: 100, checkpoint_every: 100,
        balance_sources: false, balance_labels: false, paired_sampling: false, early_stopping_patience: 0, track_coverage: true })
  end

  def write(path, value)
    EasyAI::Decision::Checkpoint.atomic_json(path, value)
  end

  def raw_rows(path)
    File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
  end

  def write_rows(path, rows)
    File.write(path, rows.map { |row| JSON.generate(row) }.join("\n") + "\n")
  end

  def verify(root)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    protocol.fetch("files_sha256").each do |name, sha|
      raise "Prepared data changed: #{name}" unless Digest::SHA256.file(File.join(root, "data", name)).hexdigest == sha
    end
    protocol
  end

  def baseline(root)
    verify(root)
    raise "Baseline diagnostics already frozen" if File.exist?(File.join(root, "baselines.json"))
    probe = raw_rows(File.join(root, "data/fit.jsonl"))
    validation = raw_rows(File.join(root, "data/validation.jsonl"))
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    results = INITIALIZERS.to_h do |name, path|
      loaded = EasyAI::Decision::Checkpoint.load(path)
      result = { "weights_sha256" => loaded.fetch(:weights_fingerprint), "fit_probe" => measure(loaded.fetch(:model), loaded.fetch(:tokenizer), probe, device),
        "validation" => measure(loaded.fetch(:model), loaded.fetch(:tokenizer), validation, device),
        "diagnostics" => diagnostics(loaded.fetch(:model), loaded.fetch(:tokenizer), probe.first(36), device) }
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      [name, result]
    end
    # Initializer selection uses the training diagnostic, never final acceptance.
    chosen = results.max_by { |_, result| result.fetch("fit_probe").fetch("factual_macro_accuracy") }.first
    loaded = EasyAI::Decision::Checkpoint.load(INITIALIZERS.fetch(chosen))
    model = EasyAI::Decision::ChoiceModel.new(config(1337))
    model.load_state_dict(loaded.fetch(:model).state_dict)
    initial = EasyAI::Decision::Checkpoint.save(File.join(root, "initial"), model: model, tokenizer: loaded.fetch(:tokenizer))
    write(File.join(root, "baselines.json"), { "models" => results, "chosen" => chosen, "initial" => initial,
      "initial_sha256" => Digest::SHA256.file(File.join(initial, "weights.pt")).hexdigest,
      "selection_basis" => "Source/language-macro factual accuracy on train-only fitting probe; no test opened" })
    puts "Initializer #{chosen}; baselines frozen"
  end

  def trainer_for(root, arm, seed, fitting: false)
    protocol = verify(root)
    baseline = JSON.parse(File.read(File.join(root, "baselines.json")))
    loaded = EasyAI::Decision::Checkpoint.load(baseline.fetch("initial"))
    raise "Initial weights changed" unless loaded.fetch(:weights_fingerprint) == baseline.fetch("initial_sha256")
    cfg = EasyAI::Decision::Config.new(protocol.fetch("config")).with(training: { seed: seed })
    cfg = cfg.with(model: { dropout: 0.0 }, training: { learning_rate: 0.0003, warmup_steps: 0 }) if fitting
    model = EasyAI::Decision::ChoiceModel.new(cfg)
    model.load_state_dict(loaded.fetch(:model).state_dict)
    rows = EasyAI::Decision::Data::Dataset.new(File.join(root, fitting ? "data/fit.jsonl" : "data/train.jsonl"))
    directory = File.join(root, fitting ? "fit" : "#{arm}-#{seed}")
    raise "Run exists; preserve interrupted evidence" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    EasyAI::Decision::FactualTrainer.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: rows, output: File.join(directory, "choice"),
      pair_margin_weight: arm == "margin" ? protocol.fetch("margin_weight") : 0.0, pair_margin: protocol.fetch("margin"))
  end

  def fit(root)
    trainer = trainer_for(root, "ce", 1337, fitting: true)
    rows = raw_rows(File.join(root, "data/fit.jsonl"))
    measurements = []
    stable_checks = 0
    (100..1000).step(100) do |budget|
      trainer.train(steps: budget)
      result = measure(trainer.model, trainer.tokenizer, rows, trainer.device).merge("step" => trainer.state.fetch("step"),
        "sampled_train_loss" => trainer.state.fetch("last_train_loss"), "device" => trainer.device)
      measurements << result
      write(File.join(root, "fit/measurements.json"), measurements)
      passed = result.fetch("accuracy") >= 0.99 && result.fetch("pair_all_correct") >= 0.95
      stable_checks = passed ? stable_checks + 1 : 0
      puts "Fitting step=#{trainer.state['step']} accuracy=#{result['accuracy']} pairs=#{result['pair_all_correct']}"
      break if stable_checks >= 2
    end
    result = measurements.last
    write(File.join(root, "fit/result.json"), result.merge("passed" => stable_checks >= 2, "consecutive_passing_checks" => stable_checks))
    raise "Fitting diagnostic failed; inspect before pilot" unless stable_checks >= 2
  end

  def train(root, arm, seed)
    raise ArgumentError, "Unknown arm" unless ARMS.include?(arm)
    raise "Fitting check has not passed" unless JSON.parse(File.read(File.join(root, "fit/result.json"))).fetch("passed")
    trainer = trainer_for(root, arm, seed)
    directory = File.join(root, "#{arm}-#{seed}")
    rows = raw_rows(File.join(root, "data/validation.jsonl"))
    probe = raw_rows(File.join(root, "data/fit.jsonl"))
    baseline = JSON.parse(File.read(File.join(root, "baselines.json")))
    reference = baseline.fetch("models").fetch("release").fetch("validation").fetch("routing_by_language")
    initial_tensor = trainer.model.named_parameters.fetch("encoder.embedding.weight").detach.cpu.clone
    best = nil
    trace = []
    trainer.train do |state, loss|
      puts "#{arm}/#{seed} step=#{state.fetch("step")} loss=#{loss.round(4)} device=#{trainer.device}" if (state.fetch("step") % 20).zero?
      next unless (state.fetch("step") % 100).zero?
      result = measure(trainer.model, trainer.tokenizer, rows, trainer.device)
      result["step"], result["device"] = state.fetch("step"), trainer.device
      result["fixed_train_probe"] = measure(trainer.model, trainer.tokenizer, probe, trainer.device)
      result["embedding_max_change"] = (trainer.model.named_parameters.fetch("encoder.embedding.weight").detach.cpu - initial_tensor).abs.max.item
      result["gpu_process_mib"] = trainer.device == "cuda" ? EasyAI::Runtime::DevicePolicy.new(requested: "cuda").process_memory_mib : nil
      result["routing_preserved"] = reference.all? { |language, cell| result.fetch("routing_by_language").fetch(language).fetch("accuracy") >= cell.fetch("accuracy") - 0.03 }
      key = [result.fetch("factual_macro_accuracy"), -result.fetch("factual_nll")]
      if result.fetch("routing_preserved") && (best.nil? || (key <=> best.fetch("key")) == 1)
        checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "selected"), model: trainer.model, tokenizer: trainer.tokenizer,
          training_state: state)
        best = { "key" => key, "checkpoint" => checkpoint, "metrics" => result }
        write(File.join(directory, "selection.json"), best)
      end
      trace << result
      write(File.join(directory, "validation.json"), trace)
      puts "#{arm}/#{seed} step=#{state['step']} loss=#{loss.round(4)} factual=#{result['factual_macro_accuracy'].round(4)} pairs=#{result['pair_all_correct'].round(4)} routing_preserved=#{result['routing_preserved']}"
    end
    write(File.join(directory, "summary.json"), { "step" => trainer.state.fetch("step"), "selected" => best,
      "device" => trainer.device, "examples_seen" => trainer.state.fetch("examples_seen"), "coverage" => trainer.state.fetch("coverage") })
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/factual-v02-r2", arm: "ce", seed: 1337 }
  OptionParser.new do |parser|
    %i[phase output arm].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "all", "prepare", "audit", "baseline", "fit", "confirm", "evaluate", "report" then DecisionFactual.public_send(options[:phase], root)
  when "train" then DecisionFactual.train(root, options[:arm], options[:seed])
  else raise ArgumentError, "Unknown factual phase"
  end
end
