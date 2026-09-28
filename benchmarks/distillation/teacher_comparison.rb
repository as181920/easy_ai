#!/usr/bin/env ruby
require "bundler/setup"
require "optparse"
require_relative "../../lib/easy_ai"

module TeacherComparison
  module_function

  def load_run(path)
    protocol = JSON.parse(File.read(File.join(path, "protocol.json")))
    data = EasyAI::Decision::Data::Dataset.new(File.join(path, "development.jsonl"))
    artifact = EasyAI::Distillation::Artifact.new(File.join(path, "teacher"))
    manifest = artifact.manifest
    unless manifest.fetch("purpose") == "development" && manifest.fetch("source_sha256") == data.fingerprint && artifact.size == data.size
      raise ArgumentError, "Expected matching development artifact"
    end
    adapter = EasyAI::Decision::DistillationAdapter.from_signature(manifest.fetch("adapter"))
    indexed = data.to_h { |row| [row.id, row] }
    originals = data.select { |row| row.id.start_with?("original:") }
    raise ArgumentError, "Expected original/reversed pairs" unless originals.size * 2 == data.size && originals.any?
    pairs = originals.map do |row|
      reverse = indexed.fetch(row.id.sub("original:", "reversed:"))
      unless reverse.to_h == row.to_h.merge("id" => reverse.id, "options" => row.options.reverse)
        raise ArgumentError, "Reversed input changed beyond candidate order"
      end
      predictions = [row, reverse].map do |example|
        record = artifact.fetch(example.id)
        unless record.fetch("identity") == adapter.identity(example) && record.dig("supervision", "kind") == "label"
          raise ArgumentError, "Expected aligned hard teacher label"
        end
        target = record.fetch("supervision").fetch("target")
        raise ArgumentError, "Teacher target missing from options" unless example.options.any? { |option| option.fetch("id") == target }
        target
      end
      { "id" => row.id, "source" => row.source, "language" => row.language, "target" => row.target,
        "original" => predictions[0], "reversed" => predictions[1] }
    end
    { "protocol" => protocol, "data_sha256" => data.fingerprint, "artifact_sha256" => artifact.fingerprint,
      "teacher" => manifest.fetch("teacher"), "adapter" => manifest.fetch("adapter"), "pairs" => pairs }
  end

  def metrics(pairs)
    count = pairs.size
    original = pairs.count { |row| row.fetch("original") == row.fetch("target") }
    reversed = pairs.count { |row| row.fetch("reversed") == row.fetch("target") }
    agreed = pairs.count { |row| row.fetch("original") == row.fetch("reversed") }
    both = pairs.count { |row| row.fetch("original") == row.fetch("target") && row.fetch("reversed") == row.fetch("target") }
    { "count" => count, "original_correct" => original, "reversed_correct" => reversed, "agreed" => agreed, "both_correct" => both,
      "original_accuracy" => original.fdiv(count), "reversed_accuracy" => reversed.fdiv(count),
      "permutation_agreement" => agreed.fdiv(count), "both_correct_rate" => both.fdiv(count) }
  end

  def summarize(run)
    sources = run.fetch("pairs").group_by { |row| row.fetch("source") }.transform_values { |rows| metrics(rows) }
    public_sources = sources.reject { |source, _| source == "relations" }
    macro = public_sources.values.sum { |value| value.fetch("original_accuracy") }.fdiv(public_sources.size)
    overall = metrics(run.fetch("pairs"))
    gate = run.fetch("protocol").fetch("gate")
    passed = sources.fetch("relations").fetch("original_accuracy") >= gate.fetch("relation_accuracy") &&
      macro >= gate.fetch("public_source_macro_accuracy") && overall.fetch("permutation_agreement") >= gate.fetch("permutation_agreement")
    run.except("pairs", "protocol").merge("overall" => overall, "by_source" => sources,
      "by_language" => run.fetch("pairs").group_by { |row| row.fetch("language") }.transform_values { |rows| metrics(rows) },
      "public_source_macro_accuracy" => macro, "passed" => passed)
  end

  def compare(baseline_path, experiment_path)
    baseline, experiment = [baseline_path, experiment_path].map { |path| load_run(path) }
    %w[protocol data_sha256 teacher].each do |key|
      raise ArgumentError, "Comparison changed #{key}; expected a prompt-only experiment" unless baseline.fetch(key) == experiment.fetch(key)
    end
    changes = baseline.fetch("pairs").zip(experiment.fetch("pairs")).filter_map do |before, after|
      raise ArgumentError, "Comparison row identity changed" unless before.except("original", "reversed") == after.except("original", "reversed")
      next if before == after
      { "id" => before.fetch("id"), "source" => before.fetch("source"), "language" => before.fetch("language"),
        "target" => before.fetch("target"), "before" => before.slice("original", "reversed"), "after" => after.slice("original", "reversed") }
    end
    transitions = %w[original reversed].to_h do |order|
      improved = changes.count { |row| row.fetch("before").fetch(order) != row.fetch("target") && row.fetch("after").fetch(order) == row.fetch("target") }
      regressed = changes.count { |row| row.fetch("before").fetch(order) == row.fetch("target") && row.fetch("after").fetch(order) != row.fetch("target") }
      [order, { "wrong_to_correct" => improved, "correct_to_wrong" => regressed }]
    end
    { "protocol" => baseline.fetch("protocol"), "baseline" => summarize(baseline), "experiment" => summarize(experiment),
      "transitions" => transitions, "changed_pairs" => changes }
  end
end

if $PROGRAM_NAME == __FILE__
  options = {}
  OptionParser.new do |parser|
    %i[baseline experiment output].each { |key| parser.on("--#{key} PATH") { |value| options[key] = value } }
  end.parse!
  report = TeacherComparison.compare(options.fetch(:baseline), options.fetch(:experiment))
  EasyAI::Distillation::Artifact.write_json(options[:output], report) if options[:output]
  puts JSON.pretty_generate(report)
end
