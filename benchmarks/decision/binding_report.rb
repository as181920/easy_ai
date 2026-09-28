#!/usr/bin/env ruby
require "json"
require "digest"
require "fileutils"
require "optparse"
require "open3"
require "cgi"
require "pathname"
require "yaml"

options = {}
OptionParser.new do |parser|
  %i[question_flip binding output].each { |key| parser.on("--#{key.to_s.tr('_', '-')} PATH") { |value| options[key] = value } }
end.parse!
output = File.expand_path(options.fetch(:output))
raise ArgumentError, "Output exists: #{output}" if File.exist?(output)
sources = %i[question_flip binding].to_h do |strategy|
  root = File.expand_path(options.fetch(strategy))
  path = File.join(root, "summary.json")
  summary = JSON.parse(File.read(path))
  raise ArgumentError, "Incomplete experiment: #{root}" unless summary["status"] == "complete"
  raise ArgumentError, "Wrong contrast strategy: #{root}" unless summary.fetch("options").fetch("contrast_strategy", "question_flip") == strategy.to_s
  [strategy, { root: root, sha256: Digest::SHA256.file(path).hexdigest, summary: summary }]
end
left, right = sources.values.map { |source| source[:summary] }
%w[seeds variants sanity_steps steps curriculum patience].each do |key|
  raise ArgumentError, "Unmatched experiment option: #{key}" unless left["options"][key] == right["options"][key]
end
raise ArgumentError, "Different datasets" unless left["data_manifest"] == right["data_manifest"]
keys = left.fetch("runs").map { |run| [run.fetch("variant"), run.fetch("seed")] }
raise ArgumentError, "Different run coverage" unless keys == right.fetch("runs").map { |run| [run.fetch("variant"), run.fetch("seed")] }
keys.each do |variant, seed|
  %w[sanity generalization].each do |phase|
    paths = sources.values.map { |source| File.join(source[:root], "#{variant}-seed-#{seed}", phase, "config.yml") }
    next unless paths.all? { |path| File.file?(path) }
    configs = paths.map do |path|
      config = YAML.safe_load_file(path)
      config.fetch("training").delete("contrast_strategy")
      config
    end
    raise ArgumentError, "Unmatched effective configuration: #{variant}/#{seed}/#{phase}" unless configs.first == configs.last
  end
end
FileUtils.mkdir_p(output)
headers = %w[strategy seed selected_step train_accuracy validation_accuracy test_accuracy test_mixed_group_all_correct zh_test en_test passed]
rows = sources.flat_map do |strategy, source|
  source[:summary].fetch("runs").map do |run|
    [strategy.to_s, run["seed"], run["selected_step"], run.dig("train", "accuracy"), run.dig("validation", "accuracy"),
      run.dig("test", "accuracy"), run.dig("test", "groups", "binding", "mixed_truth", "all_correct"),
      run.dig("test", "by_language", "zh-CN", "accuracy"), run.dig("test", "by_language", "en-US", "accuracy"),
      run.fetch("generalization_passed", false)]
  end
end
File.write(File.join(output, "comparison.tsv"), ([headers] + rows).map { |row| row.join("\t") }.join("\n") + "\n")
means = rows.group_by(&:first).transform_values do |group|
  [3, 4, 5, 6].to_h do |index|
    values = group.map { |row| row[index] }
    [headers[index], values.any?(&:nil?) ? nil : values.sum.fdiv(values.size)]
  end
end
summary = { "sources" => sources.transform_values { |source| source.slice(:root, :sha256) }, "rows" => rows.map { |row| headers.zip(row).to_h },
  "means" => means, "scope" => "Fixed split, matched-seed exploratory comparison; test is already observed. No general-semantics claim." }
File.write(File.join(output, "summary.json"), JSON.pretty_generate(summary))
plot_rows = rows.map { |row| ["#{row[0]}-#{row[1]}", *row[3..6].map { |value| value ? value * 100 : "NaN" }].join("\t") }
File.write(File.join(output, "plot.tsv"), plot_rows.join("\n") + "\n")
script = <<~GNUPLOT
  set terminal pngcairo size 1400,620 font 'Sans,11'
  set output 'comparison.png'
  set title 'Same data/model/budget; matched seeds per sampling strategy'
  set style data histograms
  set style histogram clustered gap 1
  set style fill solid 0.85 border -1
  set yrange [0:100]
  set ylabel 'Percent correct'
  set xtics rotate by -20
  set grid ytics
  set key outside top center horizontal
  plot 'plot.tsv' using 2:xtic(1) title 'Train accuracy', '' using 3 title 'Validation accuracy', '' using 4 title 'Test accuracy', '' using 5 title 'Test mixed groups: all four correct'
  set terminal svg size 1400,620 font 'Sans,11'
  set output 'comparison.svg'
  replot
GNUPLOT
File.write(File.join(output, "plot.gnuplot"), script)
_, error, status = Open3.capture3("gnuplot", stdin_data: script, chdir: output)
raise "gnuplot failed: #{error}" unless status.success?
links = sources.map do |strategy, source|
  path = Pathname.new(File.join(source[:root], "index.html")).relative_path_from(Pathname.new(output)).to_s
  "<li><a href='#{CGI.escapeHTML(path)}'>#{strategy}: per-seed metrics and original loss curves</a></li>"
end.join
File.write(File.join(output, "index.html"), "<!doctype html><meta charset='utf-8'><title>Binding comparison</title>" \
  "<h1>Binding comparison</h1><p>#{CGI.escapeHTML(summary['scope'])}</p><img src='comparison.svg' style='max-width:100%'>" \
  "<pre>#{CGI.escapeHTML(JSON.pretty_generate(means))}</pre><ul>#{links}</ul><a href='comparison.tsv'>All seed results (TSV)</a>")
puts JSON.pretty_generate(means)
