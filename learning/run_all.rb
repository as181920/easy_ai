#!/usr/bin/env ruby
require "bundler/setup"
require "optparse"
require "open3"
require "fileutils"
require "json"

options = { steps: 60, seed: 1337, output: "runs/learning", device: "auto" }
OptionParser.new do |p|
  p.banner = "Usage: bundle exec ruby learning/run_all.rb [options]"
  p.on("--steps N", Integer) { |v| options[:steps] = v }
  p.on("--chapter NAME", "Run one chapter; default runs all in order") { |v| options[:chapter] = v }
  p.on("--seed N", Integer) { |v| options[:seed] = v }
  p.on("--output PATH") { |v| options[:output] = v }
  p.on("--cudnn-dir PATH", "Optional existing directory; links only cuDNN libraries into the run output") { |v| options[:cudnn_dir] = v }
  p.on("--device NAME") { |v| options[:device] = v }
end.parse!
raise ArgumentError, "Positive steps required" unless options[:steps] > 0
chapters = Dir.children(__dir__).select { |name| name.match?(/\A\d\d_/) && File.directory?(File.join(__dir__, name)) }.sort
if options[:chapter]
  raise ArgumentError, "Unknown chapter" unless chapters.include?(options[:chapter])
  chapters = [options[:chapter]]
end
summary = []
process_env = { "OMP_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1" }
if options[:cudnn_dir]
  libraries = Dir.glob(File.join(options[:cudnn_dir], "libcudnn*.so.9"))
  raise ArgumentError, "No cuDNN 9 libraries found" if libraries.empty?
  runtime = File.expand_path(File.join(options[:output], "runtime", "cudnn"))
  FileUtils.mkdir_p(runtime)
  libraries.each do |path|
    link = File.join(runtime, File.basename(path))
    raise ArgumentError, "Refusing to replace a regular file: #{link}" if File.exist?(link) && !File.symlink?(link)
    FileUtils.ln_sf(File.expand_path(path), link)
  end
  process_env["LD_LIBRARY_PATH"] = [runtime, ENV["LD_LIBRARY_PATH"]].compact.reject(&:empty?).join(":")
end
chapters.each do |chapter|
  output = File.join(options[:output], chapter, "default")
  FileUtils.mkdir_p(output)
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  commands = [["data.rb", "--seed", options[:seed].to_s, "--output", output],
    ["train.rb", "--seed", options[:seed].to_s, "--output", output, "--device", options[:device]]]
  # XOR uses its own error-target experiment and needs a larger maximum update budget.
  commands.last.concat(["--steps", (chapter == "01_basic_nn" ? 10_000 : options[:steps]).to_s])
  commands.each do |script, *args|
    log, status = Open3.capture2e(process_env,
      RbConfig.ruby, File.join(__dir__, chapter, script), *args)
    File.write(File.join(output, script.sub(".rb", ".log")), log)
    abort "#{chapter}/#{script} failed. See #{output}/#{script.sub('.rb', '.log')}\n#{log}" unless status.success?
  end
  seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  summary << { chapter: chapter, seconds: seconds.round(3), output: output }
  puts "#{chapter}: completed (#{seconds.round(2)}s)"
end
File.write(File.join(options[:output], "run-summary.json"), JSON.pretty_generate(options: options, chapters: summary) + "\n")
