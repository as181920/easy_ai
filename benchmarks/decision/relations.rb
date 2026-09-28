#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "optparse"
require "open3"
require "rbconfig"
require "securerandom"
require_relative "../../lib/easy_ai"

# Experimental orchestration stays in benchmarks; the reusable model, sampler
# and evaluator live in lib. Every GPU stage has its own process.
class RelationExperiment
  VARIANTS = {
    "all" => { pooling: "all", encoding_mode: "separate" },
    "candidate" => { pooling: "candidate", encoding_mode: "separate" },
    "joint" => { pooling: "candidate", encoding_mode: "joint", score_mode: "linear" },
    "positions" => { pooling: "all", encoding_mode: "separate", position_scale: 0.2 },
    "rotary" => { pooling: "all", encoding_mode: "separate", position_encoding: "rotary" },
    "rotary-candidate" => { pooling: "candidate", encoding_mode: "separate", position_encoding: "rotary" },
    "rotary-joint" => { pooling: "candidate", encoding_mode: "joint", score_mode: "linear", position_encoding: "rotary" }
  }.freeze

  def initialize(options)
    @options = options
    @output = File.expand_path(options[:output])
    @data = File.expand_path(options[:data])
    @config = EasyAI::Decision::Config.load(options[:config])
    raise ArgumentError, "Output exists: #{@output}" if File.exist?(@output)
    raise ArgumentError, "At least one variant and seed are required" if options[:variants].empty? || options[:seeds].empty?
    raise ArgumentError, "Unknown variants" unless (options[:variants] - VARIANTS.keys).empty?
    raise ArgumentError, "Seeds must be nonnegative" unless options[:seeds].all? { |seed| seed >= 0 }
    raise ArgumentError, "Step budgets must be positive" unless options[:steps] > 0 && options[:sanity_steps] > 0
    raise ArgumentError, "Evaluation batch size must be positive" unless options[:evaluation_batch_size] > 0
    if options[:curriculum] && options.fetch(:patience, 0) != 0
      raise ArgumentError, "Curriculum inherits sanity configuration; use --patience 0"
    end
    EasyAI::Decision::TrainingReport.check_plotter!
    FileUtils.mkdir_p(@output)
    @summary = { "status" => "running", "options" => options, "runs" => [] }
  end

  def run
    unless File.directory?(@data)
      stage(@output, "prepare", "prepare-relations", "--output", @data, "--vocab-size", "400")
    end
    manifest = JSON.parse(File.read(File.join(@data, "manifest.json")))
    raise ArgumentError, "Unexpected relation corpus version" unless manifest["version"] == EasyAI::Decision::Data::RelationCorpus::VERSION
    manifest.fetch("files_sha256").each do |file, hash|
      raise ArgumentError, "Dataset changed: #{file}" unless Digest::SHA256.file(File.join(@data, file)).hexdigest == hash
    end
    tokenizer = EasyAI::Tokenizers::Registry.load(File.join(@data, "tokenizer.json"))
    raise ArgumentError, "Tokenizer changed" unless tokenizer.fingerprint == manifest.fetch("tokenizer_fingerprint")
    @summary["data_manifest"] = manifest
    save
    @options[:seeds].each do |seed|
      @options[:variants].each do |variant|
        root = File.join(@output, "#{variant}-seed-#{seed}")
        training = { seed: seed, device: @options[:device] }
        training[:early_stopping_patience] = @options[:patience] if @options.key?(:patience)
        training[:early_stopping_patience] = 0 if @options[:curriculum]
        config = @config.with(model: VARIANTS.fetch(variant), training: training)
        sanity = train(root, "sanity", config.with(training: { early_stopping_patience: 0 }), @options[:sanity_steps])
        fit = evaluate(File.join(root, "sanity"), "fit", sanity.fetch("checkpoint"), "sanity", controls: true)
        row = { "variant" => variant, "seed" => seed, "parameters" => sanity["parameters"], "fit" => fit,
          "sanity_passed" => fit["accuracy"] >= 0.99 && fit["maximum_permutation_logit_error"] < 1e-4 }
        if row["sanity_passed"] && !@options[:sanity_only]
          initial = @options[:curriculum] ? sanity.fetch("checkpoint") : nil
          row["initialization"] = initial || "random"
          trained = train(root, "generalization", config, @options[:steps], init: initial)
          selected = trained["best_checkpoint"] || trained.fetch("checkpoint")
          row["selected_checkpoint"] = selected
          row["selected_step"] = trained["best_step"] || trained["step"]
          %w[train validation test-familiar test].each do |split|
            row[split] = evaluate(File.join(root, "generalization"), split, selected, split, controls: split == "test")
          end
          row["generalization_passed"] = row["test"]["by_language"].values.all? do |metrics|
            metrics["accuracy"] >= 0.95 && metrics["pairs"].values.all? { |pair| pair["both_correct"] >= 0.9 }
          end
        end
        @summary["runs"] << row
        save
        warn "#{variant}/#{seed}: fit=#{fit['accuracy'].round(4)}, test=#{row.dig('test', 'accuracy') || 'not run'}"
      end
    end
    @summary["status"] = "complete"
    @summary["scope"] = "Controlled synthetic task, uncalibrated probabilities. Gate failure is a recorded experimental result, never deployment approval. Paired test variants share semantic families."
    save
    puts JSON.pretty_generate(@summary)
  rescue StandardError, Interrupt => error
    @summary["status"], @summary["error"] = "failed", error.message
    save
    raise
  end

  private

  def train(root, phase, config, steps, init: nil)
    config = config.with(training: { steps: steps })
    directory = File.join(root, phase)
    FileUtils.mkdir_p(directory)
    config_path = File.join(directory, "config.yml")
    File.write(config_path, YAML.dump(config.to_h))
    initialization = init ? ["--init", init] : ["--config", config_path, "--tokenizer", File.join(@data, "tokenizer.json")]
    args = ["train", *initialization, "--data", File.join(@data, phase == "sanity" ? "sanity.jsonl" : "train.jsonl"),
      "--output", File.join(directory, "choice"), "--steps", steps.to_s, "--progress"]
    args += ["--validation", File.join(@data, "validation.jsonl")] unless phase == "sanity"
    result = stage(directory, "training", *args)
    File.write(File.join(directory, "summary.json"), JSON.pretty_generate("selected_steps" => { "choice" => result["best_step"] || result["step"] }))
    stage(directory, "report", "report", "--input", directory)
    result
  end

  def evaluate(directory, name, checkpoint, split, controls:)
    args = ["evaluate-relations", "--checkpoint", checkpoint, "--data", File.join(@data, "#{split}.jsonl"),
      "--device", @options[:device], "--batch-size", @options[:evaluation_batch_size].to_s]
    args << "--controls" if controls
    stage(directory, name, *args)
  end

  def stage(directory, name, *args)
    FileUtils.mkdir_p(directory)
    warn "#{directory.delete_prefix(@output + '/')}: #{name}"
    result, status = nil, nil
    File.open(File.join(directory, "pipeline.log"), "a") do |log|
      Open3.popen3({ "EASY_AI_LOG_PATH" => File.join(directory, "train.log") }, RbConfig.ruby,
        File.expand_path("../../bin/easy-ai", __dir__), *args) do |input, stdout, stderr, process|
        input.close
        reader = Thread.new do
          stderr.each_line do |line|
            log.write(line)
            log.flush
            warn line if line.match?(/ (?:1|\d+00)\/\d+ /) || line.start_with?("easy-ai:")
          end
        end
        result = stdout.read
        status = process.value
        reader.value
      end
    end
    File.write(File.join(directory, "#{name}.json"), result)
    raise "Stage failed: #{directory}/#{name}; see pipeline.log" unless status.success?
    JSON.parse(result)
  end

  def save
    EasyAI::Decision::Checkpoint.atomic_json(File.join(@output, "summary.json"), @summary)
    rows = @summary["runs"].map do |row|
      [row["variant"], row["seed"], percent(row.dig("fit", "accuracy")), percent(row.dig("train", "accuracy")), percent(row.dig("validation", "accuracy")),
        percent(row.dig("test-familiar", "accuracy")), percent(row.dig("test", "accuracy")),
        row["sanity_passed"], row.fetch("generalization_passed", "not run")]
    end
    headers = %w[variant seed sanity_fit full_train validation familiar_test new_style_test sanity_gate generalization_gate]
    File.write(File.join(@output, "comparison.tsv"), ([headers] + rows).map { |row| row.join("\t") }.join("\n") + "\n")
    table = ([headers] + rows).map { |row| row.map { |value| value.to_s.ljust(20) }.join(" | ") }.join("\n")
    File.write(File.join(@output, "comparison.txt"), table + "\n")
    links = @summary["runs"].map do |row|
      root = "#{row['variant']}-seed-#{row['seed']}"
      %w[sanity generalization].filter_map do |phase|
        path = "#{root}/#{phase}/report/index.html"
        "<li><a href='#{path}'>#{root} / #{phase}: loss and memory</a></li>" if File.file?(File.join(@output, path))
      end.join
    end.join
    File.write(File.join(@output, "index.html"), "<!doctype html><meta charset='utf-8'><title>Relation experiments</title>" \
      "<h1>Controlled relation experiments</h1><p>Status: #{CGI.escapeHTML(@summary['status'])}. Fit is training accuracy; test variants share families. " \
      "Uncalibrated, synthetic explicit facts only.</p><pre>#{CGI.escapeHTML(table)}</pre><ul>#{links}</ul>")
  end

  def percent(value)
    value ? format("%.2f%%", value * 100) : "not run"
  end
end

options = { config: "config/decision/relations.yml", data: "data/decision/relations-v2",
  output: "runs/decision/relations-#{Time.now.strftime('%Y%m%d-%H%M%S')}-#{SecureRandom.hex(3)}",
  seeds: [1337, 2027, 3407], variants: %w[all candidate], sanity_steps: 1000, steps: 2000, evaluation_batch_size: 64, device: "auto" }
OptionParser.new do |parser|
  %i[config data output device].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  %i[sanity_steps steps patience evaluation_batch_size].each { |key| parser.on("--#{key.to_s.tr('_', '-')} N", Integer) { |value| options[key] = value } }
  parser.on("--seeds LIST", Array) { |values| options[:seeds] = values.map { |value| Integer(value) } }
  parser.on("--variants LIST", Array) { |values| options[:variants] = values }
  parser.on("--sanity-only") { options[:sanity_only] = true }
  parser.on("--curriculum", "Initialize full training from this run's fitted sanity weights; no early stopping") { options[:curriculum] = true }
end.parse!
RelationExperiment.new(options).run
