require "open3"
require "rbconfig"
require "securerandom"
require "time"
require "json"
require "fileutils"
require "yaml"

module EasyAI
  module Decision
    class Pipeline
      ROOT = File.expand_path("../../..", __dir__)
      DATA_FILES = %w[train validation calibration test corpus corpus-validation].freeze

      def initialize(options = {}, progress: $stderr)
        @options, @progress = options, progress
        @output = File.expand_path(options.fetch(:output) { "runs/decision/#{Time.now.strftime('%Y%m%d-%H%M%S')}-#{SecureRandom.hex(3)}" })
        @device = options.fetch(:device, "auto")
        @mlm_steps = options.fetch(:mlm_steps, 100)
        @choice_steps = options.fetch(:choice_steps, 300)
        raise ArgumentError, "MLM steps must be nonnegative" unless @mlm_steps.is_a?(Integer) && @mlm_steps >= 0
        raise ArgumentError, "Choice steps must be positive" unless @choice_steps.is_a?(Integer) && @choice_steps > 0
        if options[:resume] || options[:init] || options[:steps]
          raise ArgumentError, "Pipeline starts a new run; use --mlm-steps/--choice-steps. Resume individual stages with pretrain/train --resume."
        end
        raise ArgumentError, "Use --full-data without --limit/--train-limit" if options[:full_data] && (options[:limit] || options[:train_limit])
        @config = options[:config] ? Config.load(options[:config]) : Config.load(File.join(ROOT, "config/decision/small.yml"))
        interval = options.fetch(:eval_every) do
          budgets = [@mlm_steps, @choice_steps].select(&:positive?)
          [@config[:training]["eval_every"], *budgets.map { |steps| [steps / 10, 1].max }].min
        end
        @config = @config.with(training: { device: @device, eval_every: interval })
      end

      def run
        TrainingReport.check_plotter!
        raise ArgumentError, "Pipeline output already exists: #{@output}" if File.exist?(@output)
        FileUtils.mkdir_p(File.join(@output, "stage-results"))
        File.write(File.join(@output, "config.yml"), YAML.dump(@config.to_h))
        @started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        @summary = { "status" => "running", "output" => @output, "created_at" => Time.now.utc.iso8601,
                     "config" => @config.to_h, "mlm_steps" => @mlm_steps, "choice_steps" => @choice_steps, "stages" => [] }
        begin
          @log = File.open(File.join(@output, "pipeline.log"), "a")
          message("Run: #{@output}")
          prepare_data
          prepare_tokenizer
          validate_inputs
          train_stages
          evaluate_stages
          @summary["status"] = "complete"
          @summary["elapsed_seconds"] = Process.clock_gettime(Process::CLOCK_MONOTONIC) - @started
          save_summary
          report = TrainingReport.new(@output, out: @progress).write
          @summary["report"] = report
          @summary["active_stage"] = nil
          @summary["elapsed_seconds"] = Process.clock_gettime(Process::CLOCK_MONOTONIC) - @started
          save_summary
          message("Complete: #{report.fetch('html')}")
          @summary
        rescue StandardError, Interrupt => error
          @summary["status"] = "failed"
          @summary["error"] = "#{error.class}: #{error.message}"
          save_summary
          message("Failed; saved artifacts remain in #{@output}")
          raise
        ensure
          @log&.close
        end
      end

      private

      def prepare_data
        if @options[:data]
          @data = File.expand_path(@options[:data])
        else
          archive = File.expand_path(@options.fetch(:archive, "data/decision/downloads/massive-1.1.tar.gz"))
          stage("download", "download", "--output", archive) unless File.file?(archive)
          @data = File.join(@output, "data")
          args = ["prepare", "--archive", archive, "--output", @data,
                  "--candidates", @options.fetch(:candidates, 8).to_s, "--seed", @options.fetch(:seed, 1337).to_s]
          args += ["--limit", @options.fetch(:limit, 200).to_s] unless @options[:full_data]
          args += ["--train-limit", @options[:train_limit].to_s] if @options[:train_limit]
          args += ["--locales", @options[:locales].join(",")] if @options[:locales]
          args += ["--descriptions", File.expand_path(@options[:descriptions])] if @options[:descriptions]
          stage("prepare", *args)
        end
        missing = DATA_FILES.reject { |name| File.file?(data_file(name)) }
        raise ArgumentError, "Missing dataset files in #{@data}: #{missing.join(', ')}" unless missing.empty?
        choices = %w[train validation calibration test].to_h { |name| [name, Data::Dataset.new(data_file(name))] }
        Data::Dataset.assert_disjoint!(*choices.values)
        @summary["data"] = choices.to_h do |name, dataset|
          [name, { "rows" => dataset.size, "groups" => dataset.groups.size, "sha256" => dataset.fingerprint }]
        end
        @summary["data_directory"] = @data
        message("Data: #{@summary['data'].map { |name, info| "#{name}=#{info['rows']} rows/#{info['groups']} groups" }.join(', ')}")
      end

      def prepare_tokenizer
        if @options[:tokenizer]
          @tokenizer = File.expand_path(@options[:tokenizer])
        else
          @tokenizer = File.join(@output, "tokenizer.json")
          stage("tokenizer", "tokenizer", "--data", data_file("train"), "--output", @tokenizer,
            "--backend", @options.fetch(:backend, "ruby"), "--vocab-size", @options.fetch(:vocab_size, @config[:model]["vocab_size"]).to_s)
        end
        @summary["tokenizer"] = @tokenizer
      end

      def validate_inputs
        tokenizer = Tokenizers::Registry.load(@tokenizer)
        raise ArgumentError, "Tokenizer exceeds configured vocabulary" if tokenizer.vocab_size > @config[:model]["vocab_size"]
        collator = Data::Collator.new(tokenizer: tokenizer, config: @config)
        %w[train validation calibration test].each do |split|
          Data::Dataset.new(data_file(split)).each do |example|
            collator.state_tokens(example.state)
            example.options.each { |option| collator.option_tokens(example.question, option.fetch("text")) }
          end
        end
        mlm = Data::Dataset.new(data_file("corpus"), kind: :mlm)
        validation = Data::Dataset.new(data_file("corpus-validation"), kind: :mlm)
        groups = %w[train validation calibration test].to_h { |split| [split, Data::Dataset.new(data_file(split)).groups] }
        if (mlm.groups & (validation.groups | groups["validation"] | groups["calibration"] | groups["test"])).any? ||
            (validation.groups & (groups["train"] | groups["calibration"] | groups["test"])).any?
          raise ArgumentError, "MLM corpus groups overlap another split"
        end
        @summary["preflight_truncated_segments"] = collator.truncated
        message("Input checks passed; tokenizer=#{tokenizer.vocab_size} tokens, truncated segments=#{collator.truncated}")
      end

      def train_stages
        initialization = ["--config", File.join(@output, "config.yml"), "--tokenizer", @tokenizer]
        mlm, mlm_checkpoint = nil, nil
        if @mlm_steps > 0
          mlm = stage("mlm", "pretrain", *initialization,
            "--data", data_file("corpus"), "--validation", data_file("corpus-validation"),
            "--output", File.join(@output, "mlm"), "--steps", @mlm_steps.to_s, "--progress")
          mlm_checkpoint = mlm["best_checkpoint"] || mlm.fetch("checkpoint")
          initialization = ["--init", mlm_checkpoint]
        end
        choice = stage("choice", "train", *initialization,
          "--data", data_file("train"), "--validation", data_file("validation"),
          "--output", File.join(@output, "choice"), "--steps", @choice_steps.to_s, "--device", @device, "--progress")
        @selected_choice = choice["best_checkpoint"] || choice.fetch("checkpoint")
        @summary["selection"] = { "mlm" => mlm_checkpoint, "choice" => @selected_choice }
        @summary["selected_steps"] = { "mlm" => mlm && (mlm["best_step"] || mlm["step"]), "choice" => choice["best_step"] || choice["step"] }
        save_summary
      end

      def evaluate_stages
        if @options[:semantic_diagnostics]
          stage("semantic-diagnostic", "diagnose-semantics", "--checkpoint", @selected_choice, "--data", data_file("validation"),
            "--reference-data", data_file("train"), "--device", @device)
        end
        stage("diagnostic", "diagnose", "--checkpoint", @selected_choice, "--data", data_file("validation"),
          "--reference-data", data_file("train"), "--device", @device)
        stage("calibration", "calibrate", "--checkpoint", @selected_choice,
          "--data", data_file("calibration"), "--output", File.join(@output, "calibrated"), "--device", @device)
        stage("test", "evaluate", "--checkpoint", File.join(@output, "calibrated"), "--data", data_file("test"), "--device", @device)
      end

      def stage(name, *args)
        message("Stage #{name}: starting")
        @summary["active_stage"] = name
        save_summary
        env = {
          "OMP_NUM_THREADS" => ENV.fetch("OMP_NUM_THREADS", "1"),
          "MKL_NUM_THREADS" => ENV.fetch("MKL_NUM_THREADS", "1"),
          "EASY_AI_LOG_PATH" => File.join(@output, "train.log")
        }
        command = [RbConfig.ruby, File.join(ROOT, "bin/easy-ai"), *args]
        result, status = nil, nil
        Open3.popen3(env, *command) do |stdin, stdout, stderr, process|
          stdin.close
          reader = Thread.new { stderr.each_line { |line| message(line.chomp) } }
          result = stdout.read
          status = process.value
          reader.value
        end
        File.write(File.join(@output, "stage-results", "#{name}.json"), result)
        raise ArgumentError, "Pipeline stage #{name} failed (exit #{status.exitstatus}); see #{@output}/pipeline.log" unless status.success?
        parsed = JSON.parse(result)
        @summary["stages"] << name
        save_summary
        message("Stage #{name}: complete")
        parsed
      end

      def data_file(name)
        File.join(@data, "#{name}.jsonl")
      end

      def save_summary
        Checkpoint.atomic_json(File.join(@output, "summary.json"), @summary)
      end

      def message(text)
        @progress.puts(text)
        @progress.flush if @progress.respond_to?(:flush)
        @log&.puts(text)
        @log&.flush
      end
    end
  end
end
