#!/usr/bin/env ruby
require "bundler/setup"
require "json"
require "fileutils"

if ARGV.delete("--coverage")
  require "coverage"
  Coverage.start(lines: true)
  # Register before minitest so its autorun executes first (at_exit is LIFO).
  at_exit do
    coverage = Coverage.result
    rows = coverage.filter_map do |path, value|
      next unless path.include?("/learning/lib/easy_ai_learning/")
      next if path.end_with?("/experiment.rb") || path.include?("/course/") || path.end_with?("logic_figure.rb", "logic_report.rb")
      lines = value.fetch(:lines).compact
      next if lines.empty?
      covered = lines.count { |count| count > 0 }
      { path: path.split("/learning/").last, covered: covered, executable: lines.size, percent: (100.0 * covered / lines.size).round(2) }
    end
    FileUtils.mkdir_p("tmp/learning")
    File.write("tmp/learning/coverage.json", JSON.pretty_generate(rows) + "\n")
    covered, total = rows.sum { |r| r[:covered] }, rows.sum { |r| r[:executable] }
    puts "Core line coverage (diagnostic, not a correctness proof): #{covered}/#{total} (#{(100.0 * covered / total).round(2)}%)"
  end
end
require_relative "lib/easy_ai_learning"
EasyAILearning.loader.eager_load
Dir.glob(File.join(__dir__, "test/**/*_test.rb")).sort.each { |path| require path }
