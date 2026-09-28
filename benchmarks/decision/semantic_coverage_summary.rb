#!/usr/bin/env ruby
# Descriptive analysis of completed runs; the frequency baseline never enters inference.
require "json"

root = ARGV.fetch(0, "runs/decision/semantic-coverage-v2")
comparison = JSON.parse(File.read("#{root}/comparison.json"))
groups = comparison.fetch("results").group_by { |row| [row.fetch("arm"), row.fetch("budget")] }
summary = groups.to_h do |key, rows|
  mean = ->(&block) { rows.sum(&block).fdiv(rows.size) }
  [key.join("/"), {
    macro_accuracy: mean.call { |row| row.dig("challenge", "macro_source_accuracy") },
    nll: mean.call { |row| row.dig("challenge", "nll") },
    train_probe_accuracy: mean.call { |row| row.dig("train_probe", "macro_source_accuracy") },
    validation_accuracy: mean.call { |row| row.dig("validation", "macro_source_accuracy") },
    sources: rows.first.dig("challenge", "by_source").keys.to_h do |source|
      [source, mean.call { |row| row.dig("challenge", "by_source", source, "accuracy") }]
    end,
    permutation_agreement_range: rows.map { |row| row.dig("challenge", "permutation_agreement") }.minmax,
    selected_steps: rows.to_h { |row| [row.fetch("seed"), row.fetch("selected_step")] },
    selected_row_coverage: rows.to_h { |row| [row.fetch("seed"), row.dig("selected_checkpoint_coverage", "strata", "all", "row_coverage")] }
  }]
end
training_counts = Hash.new { |hash, source| hash[source] = Hash.new(0) }
File.foreach("#{root}/data/train.jsonl") do |line|
  row = JSON.parse(line)
  training_counts[row.fetch("source")][row.fetch("target")] += 1
end
prior = training_counts.transform_values { |labels| labels.max_by { |_label, count| count }.first }
counts = Hash.new { |hash, source| hash[source] = [0, 0] }
File.foreach("#{root}/data/challenge.jsonl") do |line|
  row = JSON.parse(line)
  counts[row.fetch("source")][0] += 1
  counts[row.fetch("source")][1] += 1 if prior.fetch(row.fetch("source")) == row.fetch("target")
end
source_accuracy = counts.transform_values { |total, correct| correct.fdiv(total) }
result = {
  groups: summary,
  source_majority_diagnostic: { labels_from_training_only: prior, source_accuracy: source_accuracy,
    macro_accuracy: source_accuracy.values.sum.fdiv(source_accuracy.size),
    scope: "Non-neural diagnostic using source metadata and training label frequencies; not an inference implementation." },
  decisions: comparison.slice("longer_training", "wording_expansion"),
  resources: comparison.fetch("resources")
}
File.write("#{root}/analysis-summary.json", JSON.pretty_generate(result) + "\n")
puts JSON.pretty_generate(result)
