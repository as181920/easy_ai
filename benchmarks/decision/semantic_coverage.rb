#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "optparse"
require "fileutils"
require "time"
require_relative "../../lib/easy_ai"

# Fixed, gold-only experiment. Public dev beyond the previously used group cap is
# reserved before fitting. The final challenge is opened only by the report phase.
module SemanticCoverage
  ROOT = File.expand_path("../..", __dir__)
  module_function

  def write_json(path, object)
    File.write("#{path}.part", JSON.pretty_generate(object) + "\n")
    File.rename("#{path}.part", path)
  end

  def config(seed)
    EasyAI::Decision::Config.load(File.join(ROOT, "config/decision/semantic.yml"))
      .with(training: { seed: seed, steps: 4000, eval_every: 200, checkpoint_every: 500,
        early_stopping_patience: 0, track_coverage: true })
  end

  def prepare(root)
    raise ArgumentError, "Output already exists" if File.exist?(root)
    data = File.join(root, "data")
    FileUtils.mkdir_p(data)
    original = File.join(ROOT, "data/decision/semantic-public")
    %w[train validation calibration tokenizer].each do |name|
      suffix = name == "tokenizer" ? "json" : "jsonl"
      FileUtils.cp(File.join(original, "#{name}.#{suffix}"), File.join(data, "#{name}.#{suffix}"))
    end
    tokenizer = EasyAI::Tokenizers::Registry.load(File.join(data, "tokenizer.json"))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config(1337))
    train = EasyAI::Decision::Data::Dataset.new(File.join(data, "train.jsonl"))
    rows = train.to_a
    counts, skipped = Hash.new(0), Hash.new(0)
    augmented = []
    rows.sort_by { |row| Digest::SHA256.hexdigest("coverage-v1:#{row.id}") }.each do |row|
      collator.state_tokens(row.state)
      row.options.each { |option| collator.option_tokens(row.question, option.fetch("text")) }
      key = "#{row.source}/#{row.language}/#{row.target}"
      next if counts[key] >= 3000
      changed = EasyAI::Decision::Data::SemanticExpansion.call(row)
      begin
        changed.fetch("options").each { |option| collator.option_tokens(changed.fetch("question"), option.fetch("text")) }
      rescue ArgumentError => error
        raise unless error.message.include?("exceeds")
        skipped[key] += 1
        next
      end
      augmented << changed
      counts[key] += 1
    end
    File.open(File.join(data, "expanded.jsonl"), "w") do |file|
      rows.each { |row| file.puts(JSON.generate(row.to_h)) }
      augmented.each { |row| file.puts(JSON.generate(row)) }
    end
    challenge = prepare_challenge(original, collator)
    File.write(File.join(data, "challenge.jsonl"), challenge.map { |row| JSON.generate(row) }.join("\n") + "\n")
    splits = %w[train validation calibration challenge].map { |name| EasyAI::Decision::Data::Dataset.new(File.join(data, "#{name}.jsonl")) }
    EasyAI::Decision::Data::Dataset.assert_disjoint!(*splits)
    expanded = EasyAI::Decision::Data::Dataset.new(File.join(data, "expanded.jsonl"))
    EasyAI::Decision::Data::Dataset.assert_disjoint!(expanded, *splits.drop(1))
    protocol = { "version" => 1, "created_at" => Time.now.utc.iso8601, "seeds" => [1337, 2027, 3407],
      "budgets" => [1000, 4000], "arms" => %w[baseline expanded], "config" => config(1337).to_h,
      "expansion_version" => EasyAI::Decision::Data::SemanticExpansion::VERSION,
      "expansion_counts" => counts, "expansion_over_length_skipped" => skipped,
      "tokenizer_sha256" => Digest::SHA256.file(File.join(data, "tokenizer.json")).hexdigest,
      "source_sha256" => Dir.glob(File.join(ROOT, "data/decision/downloads/semantics/*")).select { |path| File.file?(path) }
        .to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
      "files_sha256" => Dir.glob(File.join(data, "*.jsonl")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
      "gate" => { "mean_macro_accuracy_gain" => 0.03, "maximum_mean_source_regression" => 0.03, "minimum_improved_seeds" => 2 },
      "scope" => "Fixed 4000-update cap, all seeds retained; select checkpoints by existing validation NLL only. Fresh public dev challenge is final evaluation, not private novel data. Expansion adds expression variants, not independent facts or source groups. Select by declared source/language/gold label, never by keywords." }
    write_json(File.join(root, "protocol.json"), protocol)
    %w[train expanded challenge].each do |name|
      write_json(File.join(root, "#{name}-audit.json"), EasyAI::Decision::Data::CoverageAudit.call(
        EasyAI::Decision::Data::Dataset.new(File.join(data, "#{name}.jsonl"))))
    end
    puts JSON.pretty_generate(protocol)
  end

  def prepare_challenge(original, collator)
    used = %w[train validation calibration test].flat_map do |name|
      EasyAI::Decision::Data::Dataset.new(File.join(original, "#{name}.jsonl")).groups.to_a
    end.to_set
    raw = EasyAI::Decision::Data::SemanticAdapter.each(File.join(ROOT, "data/decision/downloads/semantics")).to_a
    eligible = EasyAI::Decision::Data::SemanticCorpus.new(raw).split_rows.select { |_row, group, split| split == "test" && !used.include?(group) }
    selected, seen = Hash.new { |hash, key| hash[key] = Set.new }, Set.new
    eligible.sort_by { |row, group, _split| [row[:source], Digest::SHA256.hexdigest("fresh-coverage-v1:#{group}"), row[:question]] }.filter_map do |row, group, _split|
      groups = selected[row.fetch(:source)]
      next if groups.size >= 150 && !groups.include?(group)
      identity = Digest::SHA256.hexdigest([row[:source], row[:state], row[:question], row[:target]].join("\n"))
      next unless seen.add?(identity)
      options = row.fetch(:options).map { |id, text| { "id" => id, "text" => text } }.shuffle(random: Random.new(identity.to_i(16)))
      begin
        collator.state_tokens(row.fetch(:state))
        options.each { |option| collator.option_tokens(row.fetch(:question), option.fetch("text")) }
      rescue ArgumentError => error
        raise unless error.message.include?("exceeds")
        next
      end
      groups << group
      { "id" => identity, "group_id" => group, "language" => row.fetch(:language), "source" => row.fetch(:source),
        "state" => row.fetch(:state), "question" => row.fetch(:question), "options" => options, "target" => row.fetch(:target) }
    end
  end

  def train(root, arm, seed)
    GC.start
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    raise ArgumentError, "Unknown arm/seed" unless protocol.fetch("arms").include?(arm) && protocol.fetch("seeds").include?(seed)
    directory = File.join(root, "#{arm}-#{seed}")
    raise ArgumentError, "Run already exists; preserve completed/interrupted evidence" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    File.symlink("train.log", File.join(directory, "pipeline.log"))
    cfg = EasyAI::Decision::Config.new(protocol.fetch("config")).with(training: { seed: seed })
    File.write(File.join(directory, "config.yml"), cfg.to_h.to_yaml)
    data_path = File.join(root, "data", arm == "baseline" ? "train.jsonl" : "expanded.jsonl")
    dataset = EasyAI::Decision::Data::Dataset.new(data_path)
    raise ArgumentError, "Training data changed" unless dataset.fingerprint == protocol.fetch("files_sha256").fetch(File.basename(data_path))
    validation = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/validation.jsonl"))
    raise ArgumentError, "Validation data changed" unless validation.fingerprint == protocol.fetch("files_sha256").fetch("validation.jsonl")
    unless Digest::SHA256.file(File.join(root, "data/tokenizer.json")).hexdigest == protocol.fetch("tokenizer_sha256")
      raise ArgumentError, "Tokenizer changed"
    end
    tokenizer = EasyAI::Tokenizers::Registry.load(File.join(root, "data/tokenizer.json"))
    Torch.manual_seed(seed)
    trainer = EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(cfg), tokenizer: tokenizer,
      dataset: dataset, validation: validation, output: File.join(directory, "choice"))
    progress = EasyAI::Decision::Progress.new(task: "#{arm}/#{seed}", total: cfg[:training]["steps"])
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    summaries = []
    protocol.fetch("budgets").each do |budget|
      trainer.train(steps: budget) { |state, loss| progress.update(state, loss, device: trainer.device) }
      selected = EasyAI::Decision::Checkpoint.resolve(File.join(directory, "choice/best"))
      destination = File.join(directory, "selected-#{budget}")
      FileUtils.cp_r(selected, destination)
      coverage = trainer.state.fetch("coverage")
      write_json(File.join(directory, "coverage-#{budget}.json"), EasyAI::Decision::Data::CoverageAudit.call(dataset,
        visits: coverage.fetch("row_visits"), input_tokens: coverage.fetch("input_tokens")))
      summaries << { "budget" => budget, "selected_step" => trainer.state.fetch("best_step"), "checkpoint" => File.expand_path(destination),
        "best_validation_loss" => trainer.state.fetch("best_validation_loss"), "elapsed_seconds" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - started,
        "device" => trainer.device, "examples_seen" => trainer.state.fetch("examples_seen"), "input_tokens" => coverage.fetch("input_tokens") }
      write_json(File.join(directory, "summary.json"), { "arm" => arm, "seed" => seed, "budgets" => summaries,
        "selected_steps" => { "choice" => trainer.state.fetch("best_step") } })
      EasyAI::Decision::TrainingReport.new(directory).write
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/semantic-coverage-v2", seed: 1337, arm: "baseline", budget: 4000 }
  OptionParser.new do |parser|
    %i[phase output arm].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
    parser.on("--budget N", Integer) { |value| options[:budget] = value }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "prepare" then SemanticCoverage.prepare(root)
  when "train" then SemanticCoverage.train(root, options[:arm], options[:seed])
  when "evaluate"
    require_relative "semantic_coverage_report"
    SemanticCoverageReport.evaluate(root, options[:arm], options[:seed], options[:budget])
  when "report"
    require_relative "semantic_coverage_report"
    puts JSON.pretty_generate(SemanticCoverageReport.write(root))
  when "all"
    SemanticCoverage.prepare(root)
    %w[baseline expanded].each do |arm|
      [1337, 2027, 3407].each { |seed| SemanticCoverage.train(root, arm, seed) }
    end
    require_relative "semantic_coverage_report"
    %w[baseline expanded].product([1337, 2027, 3407], [1000, 4000]).each do |arm, seed, budget|
      SemanticCoverageReport.evaluate(root, arm, seed, budget)
      GC.start
    end
    puts JSON.pretty_generate(SemanticCoverageReport.write(root))
  else raise ArgumentError, "Expected --phase prepare, train, evaluate, report or all"
  end
end
