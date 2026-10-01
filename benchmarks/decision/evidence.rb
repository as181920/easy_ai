#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require_relative "semantic_coverage"
require "rbconfig"

# Frozen paired comparison; each subprocess releases all CUDA tensors on exit.
module EvidenceExperiment
  ROOT = SemanticCoverage::ROOT
  SEEDS = [1337, 2027, 3407].freeze
  ARMS = %w[answer evidence].freeze
  module_function

  def config(seed, arm)
    SemanticCoverage.config(seed).with(model: { evidence_head: true }, training: {
      steps: 800, learning_rate: 0.0001, warmup_steps: 50, eval_every: 100,
      checkpoint_every: 200, evidence_loss_weight: arm == "evidence" ? 0.2 : 0.0 })
  end

  def parent(seed)
    File.join(ROOT, "runs/decision/semantic-coverage-v2/baseline-#{seed}/selected-4000")
  end

  def prepare(root)
    raise ArgumentError, "Output already exists" if File.exist?(root)
    FileUtils.mkdir_p(root)
    data = File.join(root, "data")
    EasyAI::Decision::Data::EvidenceCorpus.new.write(output: data)
    original = File.join(ROOT, "data/decision/semantic-public")
    FileUtils.cp(File.join(original, "tokenizer.json"), File.join(data, "tokenizer.json"))
    tokenizer = EasyAI::Tokenizers::Registry.load(File.join(data, "tokenizer.json"))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config(1337, "answer"))
    %w[train validation calibration test test-familiar sanity].each do |split|
      dataset = EasyAI::Decision::Data::Dataset.new(File.join(data, "#{split}.jsonl"))
      dataset.each_slice(16) { |batch| collator.call(batch, with_evidence: true); GC.start }
    end
    File.open(File.join(data, "mixed.jsonl"), "w") do |file|
      [File.join(original, "train.jsonl"), File.join(data, "train.jsonl")].each do |path|
        File.foreach(path) { |line| file.write(line) }
      end
    end
    validation = [File.join(original, "validation.jsonl"), File.join(data, "validation.jsonl")].flat_map do |path|
      EasyAI::Decision::Data::Dataset.new(path).to_a
    end.group_by(&:source).values.flat_map do |rows|
      rows.sort_by { |row| Digest::SHA256.hexdigest("evidence-selection:#{row.id}") }.first(96)
    end
    write_rows(File.join(data, "selection.jsonl"), validation.map(&:to_h))
    # Exclude the preceding round's challenge too; no reusing its evaluation groups.
    previous = EasyAI::Decision::Data::Dataset.new(File.join(ROOT, "runs/decision/semantic-coverage-v2/data/challenge.jsonl")).groups
    used = %w[train validation calibration test].flat_map do |split|
      EasyAI::Decision::Data::Dataset.new(File.join(original, "#{split}.jsonl")).groups.to_a
    end.to_set | previous
    raw = EasyAI::Decision::Data::SemanticAdapter.each(File.join(ROOT, "data/decision/downloads/semantics")).to_a
    selected = Hash.new { |hash, key| hash[key] = Set.new }
    challenge = EasyAI::Decision::Data::SemanticCorpus.new(raw).split_rows
      .sort_by { |row, group, _split| [row[:source], Digest::SHA256.hexdigest("evidence-public:#{group}"), row[:question]] }
      .filter_map do |row, group, split|
        next unless split == "test" && !used.include?(group)
        groups = selected[row[:source]]
        next if groups.size >= 150 && !groups.include?(group)
        id = Digest::SHA256.hexdigest([row[:source], row[:state], row[:question], row[:target]].join("\n"))
        options = row[:options].map { |key, text| { "id" => key, "text" => text } }.shuffle(random: Random.new(id.to_i(16)))
        begin
          collator.state_tokens(row[:state])
          options.each { |option| collator.option_tokens(row[:question], option["text"]) }
        rescue ArgumentError => error
          raise unless error.message.include?("exceeds")
          next
        end
        groups << group
        { "id" => id, "group_id" => group, "source" => row[:source], "language" => row[:language],
          "state" => row[:state], "question" => row[:question], "options" => options, "target" => row[:target] }
      end.uniq { |row| row["id"] }
    write_rows(File.join(data, "public-test.jsonl"), challenge)
    files = Dir.glob(File.join(data, "*.json*")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] }
    parents = SEEDS.to_h { |seed| [seed, Digest::SHA256.file(File.join(parent(seed), "weights.pt")).hexdigest] }
    SemanticCoverage.write_json(File.join(root, "protocol.json"), {
      "seeds" => SEEDS, "arms" => ARMS, "steps" => 800, "sanity_steps" => 300,
      "config" => config(1337, "answer").to_h, "configs" => ARMS.to_h { |arm| [arm, config(1337, arm).to_h] }, "files_sha256" => files, "parents_sha256" => parents,
      "gate" => { "mean_group_gain" => 0.05, "minimum_improved_seeds" => 2,
        "maximum_language_regression" => 0.03, "maximum_public_regression" => 0.03,
        "probability_policy" => "Mean calibrated controlled and public NLL must not worsen against answer-only",
        "readiness_language_accuracy" => 0.95, "readiness_binding_all_correct" => 0.90 },
      "scope" => "Controlled generated Chinese/English explicit facts. 32 independent held-out families, 6144 rows; familiar test shares those families. Validation NLL selects checkpoints. All budgets and seeds retained. Public replay is shared; no teacher, tokenizer fitting, inference rules or test-driven tuning. Sanity is a capacity diagnostic only." })
    puts "Prepared #{root}; controlled test=6144, public test=#{challenge.size}"
  end

  def write_rows(path, rows)
    File.open(path, "w") { |file| rows.each { |row| file.puts(JSON.generate(row)) } }
  end

  def verify(root)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    protocol.fetch("files_sha256").each do |name, digest|
      raise ArgumentError, "Prepared data changed: #{name}" unless Digest::SHA256.file(File.join(root, "data", name)).hexdigest == digest
    end
    protocol
  end

  def train(root, arm, seed, sanity: false)
    protocol = verify(root)
    raise ArgumentError, "Unknown arm/seed" unless ARMS.include?(arm) && SEEDS.include?(seed)
    source = EasyAI::Decision::Checkpoint.load(parent(seed))
    raise ArgumentError, "Parent weights changed" unless source.fetch(:weights_fingerprint) == protocol.fetch("parents_sha256").fetch(seed.to_s)
    cfg = EasyAI::Decision::Config.new(protocol.fetch("configs").fetch(arm)).with(training: { seed: seed })
    cfg = cfg.with(training: { steps: 300, eval_every: 50, balance_sources: false }) if sanity
    Torch.manual_seed(seed)
    model = EasyAI::Decision::ChoiceModel.new(cfg)
    Torch.no_grad { source.fetch(:model).state_dict.each { |name, tensor| model.state_dict.fetch(name).copy!(tensor) } }
    source[:model] = nil
    GC.start
    name = sanity ? "sanity-#{arm}-#{seed}" : "#{arm}-#{seed}"
    directory = File.join(root, name)
    raise ArgumentError, "Run exists" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    File.symlink("train.log", File.join(directory, "pipeline.log"))
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data", sanity ? "sanity.jsonl" : "mixed.jsonl"))
    validation = EasyAI::Decision::Data::Dataset.new(File.join(root, "data", sanity ? "sanity.jsonl" : "selection.jsonl"))
    trainer = EasyAI::Decision::Trainer.new(model: model, tokenizer: source.fetch(:tokenizer),
      dataset: dataset, validation: sanity ? nil : validation, output: File.join(directory, "choice"))
    progress = EasyAI::Decision::Progress.new(task: name, total: cfg[:training]["steps"])
    trainer.train { |state, loss| progress.update(state, loss, device: trainer.device) }
    selected = sanity ? trainer.last_checkpoint : EasyAI::Decision::Checkpoint.resolve(File.join(directory, "choice/best"))
    FileUtils.cp_r(selected, File.join(directory, "selected"))
    SemanticCoverage.write_json(File.join(directory, "summary.json"), { "arm" => arm, "seed" => seed,
      "step" => trainer.state["step"], "selected_steps" => { "choice" => trainer.state["best_step"] || trainer.state["step"] },
      "device" => trainer.device, "best_validation_loss" => trainer.state["best_validation_loss"],
      "examples_seen" => trainer.state["examples_seen"] })
    EasyAI::Decision::TrainingReport.new(directory).write
  end

  def subprocess(root, phase, arm, seed)
    command = [RbConfig.ruby, __FILE__, "--output", root, "--phase", phase, "--arm", arm, "--seed", seed.to_s]
    raise "Subprocess failed: #{phase}/#{arm}/#{seed}" unless system(*command)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/evidence-v1", arm: "answer", seed: 1337 }
  OptionParser.new do |parser|
    %i[phase output arm].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "prepare" then EvidenceExperiment.prepare(root)
  when "train", "sanity" then EvidenceExperiment.train(root, options[:arm], options[:seed], sanity: options[:phase] == "sanity")
  when "evaluate"
    require_relative "evidence_evaluation"
    EvidenceEvaluation.run(root, options[:arm], options[:seed])
    if File.exist?(File.join(root, "generalization/manifest.json"))
      command = [RbConfig.ruby, File.join(__dir__, "generalization.rb"), "--phase", "evaluate", "--output", root,
        "--arm", options[:arm], "--seed", options[:seed].to_s]
      raise "Generalization evaluation failed" unless system(*command)
    end
  when "report"
    require_relative "evidence_evaluation"
    puts JSON.pretty_generate(EvidenceEvaluation.report(root))
  when "all"
    EvidenceExperiment.prepare(root)
    require_relative "generalization"
    DecisionGeneralization.download
    DecisionGeneralization.prepare(root)
    EvidenceExperiment::ARMS.each { |arm| EvidenceExperiment.subprocess(root, "sanity", arm, 1337) }
    EvidenceExperiment::ARMS.product(EvidenceExperiment::SEEDS).each { |arm, seed| EvidenceExperiment.subprocess(root, "train", arm, seed) }
    (%w[parent] + EvidenceExperiment::ARMS).product(EvidenceExperiment::SEEDS).each { |arm, seed| EvidenceExperiment.subprocess(root, "evaluate", arm, seed) }
    require_relative "evidence_evaluation"
    puts JSON.pretty_generate(EvidenceEvaluation.report(root))
  else raise ArgumentError, "Unknown phase"
  end
end
