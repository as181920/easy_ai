#!/usr/bin/env ruby
require_relative "release"

# One bounded delivery correction; opened acceptance never becomes a fresh test again.
module DecisionReleaseCorrective
  module_function

  def fresh_rows(observed)
    raise "MASSIVE archive changed" unless Digest::SHA256.file(DecisionRelease::ARCHIVE).hexdigest == DecisionRelease::ARCHIVE_SHA256
    raw = EasyAI::Decision::Data::NaturalAdapter.massive(DecisionRelease::ARCHIVE, partitions: %w[train dev test]).to_a
    raw.group_by { |row| row.fetch("group_id") }.values.select do |group|
      group.all? { |row| row.fetch("partition") == "train" && !observed.include?(EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state"))) }
    end.flatten
  end

  def prepare(root, previous)
    raise "Corrective run exists" if File.exist?(root)
    previous_protocol = DecisionRelease.verify(previous)
    observed, _, historical_files = DecisionRelease.history
    fresh = fresh_rows(observed)
    loaded = EasyAI::Decision::Checkpoint.load(File.join(previous, "selected"))
    config = loaded.fetch(:model).config.with(training: { steps: 800, learning_rate: 0.00005, warmup_steps: 50,
      eval_every: 100, checkpoint_every: 200, early_stopping_patience: 4, balance_labels: false })
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: config)
    inputs = DecisionRelease.rows(File.join(previous, "data/train.jsonl")).map { |row| row.to_h.merge("partition" => "train") }
    original_material = inputs.map { |row| EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state")) }.to_set
    fresh.each { |row| row["original_partition"] = row.fetch("partition"); row["partition"] = "test" }
    corpus = EasyAI::Decision::Data::RoutingCorpus.new(inputs + fresh, historical_material: observed, train_material: original_material, collator: collator)
    test = DecisionRelease.panel(corpus.splits.fetch("test"), 400)
    counts = test.group_by { |row| row.fetch("language") }.transform_values(&:size)
    raise "Not enough genuinely unused acceptance groups: #{counts}" unless DecisionRelease::LANGUAGES.all? { |language| counts.fetch(language, 0) >= 300 }
    FileUtils.mkdir_p(File.join(root, "data"))
    %w[train validation calibration].each { |name| FileUtils.cp(File.join(previous, "data/#{name}.jsonl"), File.join(root, "data/#{name}.jsonl")) }
    File.open(File.join(root, "data/test.jsonl"), "w") { |file| test.each { |row| file.puts(JSON.generate(row)) } }
    model = EasyAI::Decision::ChoiceModel.new(config)
    model.load_state_dict(loaded.fetch(:model).state_dict)
    initial = EasyAI::Decision::Checkpoint.save(File.join(root, "initial"), model: model, tokenizer: loaded.fetch(:tokenizer))
    protocol = previous_protocol.merge("steps" => 800, "config" => config.to_h, "initial" => initial,
      "initial_sha256" => Digest::SHA256.file(File.join(initial, "weights.pt")).hexdigest,
      "parent_sha256" => loaded.fetch(:weights_fingerprint), "historical_files_sha256" => historical_files,
      "previous_protocol_sha256" => Digest::SHA256.file(File.join(previous, "protocol.json")).hexdigest,
      "previous_exclusions" => previous_protocol.fetch("exclusions"), "exclusions" => corpus.exclusions,
      "calibration_guard" => { "accepted_accuracy" => 0.94, "accepted_accuracy_lower_95" => 0.90 },
      "files_sha256" => Dir.glob(File.join(root, "data/*")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
      "counts" => %w[train validation calibration test].to_h do |name|
        [name, DecisionRelease.rows(File.join(root, "data/#{name}.jsonl")).group_by(&:language).transform_values(&:size)]
      end,
      "scope" => "18 MASSIVE bilingual request domains. Bounded corrective fit: own selected weights, same train/validation/calibration; 800 updates at LR .00005 with natural within-language sampling, no class oversampling. Fresh acceptance prospectively reserved from previously unused official TRAIN groups, not the opened official-test panel; all translations/material checked against historical prepared data. Original official partition retained in test records. Conservative calibration guards .94 point/.90 Wilson lower target .90 independent selected accuracy; acceptance gates unchanged. Historical calibration reused as development, not claimed untouched. No business integration, teacher or generic semantic claim.")
    DecisionRelease.write(File.join(root, "protocol.json"), protocol)
    DecisionRelease.write(File.join(root, "evaluation-unit.json"), { "evaluation_unit" => DecisionRelease::EVALUATION_UNIT, "frozen_before_calibration_and_acceptance" => true })
    puts JSON.pretty_generate(protocol.slice("counts", "calibration_guard"))
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/release-v01-corrective", previous: "runs/decision/release-v01-routing" }
  OptionParser.new do |parser|
    %i[phase output previous].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  end.parse!
  raise "Expected all or prepare" unless %w[all prepare].include?(options[:phase])
  root = File.expand_path(options[:output])
  DecisionReleaseCorrective.prepare(root, File.expand_path(options[:previous]))
  if options[:phase] == "all"
    %w[baseline train calibrate acceptance package].each do |phase|
      raise "Corrective #{phase} failed" unless system(RbConfig.ruby, File.join(__dir__, "release.rb"), "--phase", phase, "--output", root)
    end
  end
end
