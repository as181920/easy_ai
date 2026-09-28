require "optparse"
require "json"
require "fileutils"

module EasyAI
  module Decision
    class Cli
      COMMANDS = %w[pipeline semantic-pipeline report download download-semantics prepare prepare-semantics prepare-relations prepare-corpus tokenizer pretrain train calibrate evaluate evaluate-relations diagnose diagnose-semantics predict grow inspect].freeze

      def self.run(argv, out: $stdout, err: $stderr)
        new(out: out, err: err).run(argv)
      end

      def initialize(out:, err:)
        @out, @err = out, err
      end

      def run(argv)
        command = argv.shift
        return help if !command || %w[help --help -h].include?(command)
        raise ArgumentError, "Unknown command #{command}" unless COMMANDS.include?(command)
        @options = { backend: "ruby", candidates: 8, seed: 1337, operation: "add_block" }
        parser = option_parser(command)
        parser.parse!(argv)
        return 0 if @options[:help]
        raise ArgumentError, "Unexpected arguments: #{argv.join(' ')}" unless argv.empty?
        result = public_send("command_#{command.tr('-', '_')}")
        @out.puts(JSON.pretty_generate(result))
        0
      rescue ArgumentError, KeyError, OptionParser::ParseError, Errno::ENOENT, JSON::ParserError => error
        @err.puts("easy-ai: #{error.message}")
        1
      end

      def command_download
        Data::Download.massive(required(:output), expected_sha256: @options[:sha256])
      end

      def command_download_semantics
        Data::SemanticSources.prepare(required(:output))
      end

      def command_prepare_relations
        Data::RelationCorpus.new(seed: @options[:seed]).write(output: required(:output), vocab_size: @options.fetch(:vocab_size, 400),
          sanity_families: @options.fetch(:sanity_families, 1))
      end

      def command_evaluate_relations
        predictor = Predictor.load(required(:checkpoint), device: @options.fetch(:device, "auto"))
        data = Data::Dataset.new(required(:data))
        path = Checkpoint.resolve(required(:checkpoint))
        metadata = JSON.parse(File.read(File.join(path, "metadata.json")))
        used = metadata.fetch("training", {}).fetch("groups", {}).values.flatten.to_set
        used.merge(metadata.dig("calibration", "groups") || [])
        overlap = (data.groups & used).size
        RelationEvaluation.new(predictor, batch_size: @options.fetch(:batch_size, 16)).evaluate(data, controls: @options[:controls]).merge(
          "previously_used_groups" => overlap, "scope" => overlap > 0 ? "Training/validation diagnostic; not an independent test" : "Groups not used to train/select these weights")
      end

      def command_prepare_semantics
        directory = required(:input)
        Data::SemanticSources.prepare(directory)
        config = Config.load(@options.fetch(:config, File.join(Pipeline::ROOT, "config/decision/semantic.yml")))
        rows = Data::SemanticAdapter.each(directory).to_a
        result = Data::SemanticCorpus.new(rows, seed: @options[:seed]).write(output: required(:output), config: config,
          tokenizer: Tokenizers::Registry.build("native"), limit: @options.fetch(:limit, 100), train_limit: @options[:train_limit])
        FileUtils.cp(File.join(directory, "sources.json"), File.join(required(:output), "sources.json"))
        result
      end

      def command_pipeline
        Pipeline.new(@options, progress: @err).run
      end

      def command_semantic_pipeline
        config = @options.fetch(:config, File.join(Pipeline::ROOT, "config/decision/semantic.yml"))
        data = @options.fetch(:data, "data/decision/semantic-public")
        unless File.directory?(data)
          directory = "data/decision/downloads/semantics"
          @err.puts("Preparing public semantic data: #{data}")
          Data::SemanticSources.prepare(directory)
          Data::SemanticCorpus.new(Data::SemanticAdapter.each(directory).to_a, seed: @options[:seed]).write(
            output: data, config: Config.load(config), tokenizer: Tokenizers::Registry.build("native"),
            limit: @options.fetch(:limit, 100), train_limit: @options[:train_limit])
          FileUtils.cp(File.join(directory, "sources.json"), File.join(data, "sources.json"))
        end
        options = @options.merge(config: config, data: data, tokenizer: File.join(data, "tokenizer.json"),
          mlm_steps: @options.fetch(:mlm_steps, 1000), choice_steps: @options.fetch(:choice_steps, 1000),
          eval_every: @options.fetch(:eval_every, 100), semantic_diagnostics: true)
        Pipeline.new(options, progress: @err).run
      end

      def command_report
        TrainingReport.new(required(:input), out: @err).write
      end

      def command_prepare
        Data::Adapters::Massive.prepare(archive: required(:archive), output: required(:output),
          locales: @options[:locales] || Data::Adapters::Massive::DEFAULT_LOCALES,
          candidates: @options[:candidates], limit: @options[:limit], train_limit: @options[:train_limit], seed: @options[:seed],
          descriptions: @options[:descriptions] && JSON.parse(File.read(@options[:descriptions])))
      end

      def command_prepare_corpus
        Data::Corpus.prepare(input: required(:input), output: required(:output), language: @options.fetch(:language, "und"), limit: @options[:limit])
      end

      def command_tokenizer
        source = required(:data)
        texts = Enumerator.new do |yielder|
          File.foreach(source) do |line|
            next if line.strip.empty?
            row = JSON.parse(line)
            if row.key?("text")
              yielder << row.fetch("text")
            else
              Data::Example.new(row).texts.each { |text| yielder << text }
            end
          end
        end
        tokenizer = Tokenizers::Registry.build(@options[:backend])
        start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        tokenizer.train(texts, vocab_size: @options.fetch(:vocab_size, 32_000))
        path = required(:output)
        raise ArgumentError, "Tokenizer output already exists" if File.exist?(path)
        FileUtils.mkdir_p(File.dirname(path))
        tokenizer.save(path)
        { "path" => path, "backend" => @options[:backend], "vocab_size" => tokenizer.vocab_size,
         "source_sha256" => Digest::SHA256.file(source).hexdigest,
         "fingerprint" => tokenizer.fingerprint, "seconds" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - start }
      end

      def command_pretrain
        train_task(:mlm)
      end

      def command_train
        train_task(:choice)
      end

      def command_calibrate
        loaded = Checkpoint.load(required(:checkpoint))
        data = Data::Dataset.new(required(:data))
        groups = loaded[:metadata].fetch("training").fetch("groups", {})
        used = groups.values.flatten.to_set
        raise ArgumentError, "Calibration overlaps previously used data" unless (data.groups & used).empty?
        predictor = Predictor.new(model: loaded[:model], tokenizer: loaded[:tokenizer], device: @options.fetch(:device, "auto"))
        rows = Evaluator.new(predictor).collect(data)
        calibrator = Calibrator.new.fit(rows[:logits], rows[:targets])
        calibration = { "temperature" => calibrator.temperature, "dataset_sha256" => data.fingerprint,
          "groups" => data.groups.to_a, "count" => data.size,
          "before" => Evaluator.metrics(rows[:logits], rows[:targets]),
          "after" => Evaluator.metrics(rows[:logits], rows[:targets], calibrator) }
        # Calibration artifacts are inference checkpoints, not resumable training runs.
        metadata = loaded[:metadata]["training"].merge("calibration_parent" => loaded[:weights_fingerprint])
        path = Checkpoint.save(required(:output), model: predictor.model, tokenizer: loaded[:tokenizer],
          training_state: metadata, calibration: calibration)
        { "checkpoint" => path, "calibration" => calibration.reject { |key, _| key == "groups" } }
      end

      def command_evaluate
        loaded = Checkpoint.load(required(:checkpoint))
        data = Data::Dataset.new(required(:data))
        used = loaded[:metadata].fetch("training").fetch("groups", {}).values.flatten
        used += loaded[:metadata].dig("calibration", "groups") || []
        raise ArgumentError, "Evaluation overlaps train/validation/calibration groups" unless (data.groups & used.to_set).empty?
        predictor = Predictor.load(required(:checkpoint), device: @options.fetch(:device, "auto"))
        Evaluator.new(predictor).evaluate(data).merge("dataset_sha256" => data.fingerprint,
          "calibrated" => predictor.calibrated, "device" => predictor.device)
      end

      def command_predict
        request = JSON.parse(@options[:input] ? File.read(@options[:input]) : $stdin.read)
        predictor = Predictor.load(required(:checkpoint), device: @options.fetch(:device, "auto"))
        predictor.probabilities(state: request.fetch("state"), question: request.fetch("question"), options: request.fetch("options"))
      end

      def command_diagnose
        predictor = Predictor.load(required(:checkpoint), device: @options.fetch(:device, "cpu"))
        result = StateAblation.new(predictor: predictor, validation: Data::Dataset.new(required(:data)),
          reference: Data::Dataset.new(required(:reference_data)), language: @options[:language],
          limit: @options.fetch(:limit, 200), seed: @options[:seed]).evaluate
        result.merge("checkpoint" => File.expand_path(required(:checkpoint)))
      end

      def command_diagnose_semantics
        predictor = Predictor.load(required(:checkpoint), device: @options.fetch(:device, "cpu"))
        SemanticDiagnostics.new(predictor: predictor, validation: Data::Dataset.new(required(:data)),
          reference: Data::Dataset.new(required(:reference_data)), limit: @options.fetch(:limit, 50), seed: @options[:seed]).evaluate
      end

      def command_grow
        loaded = Checkpoint.load(required(:checkpoint))
        model = case @options[:operation]
                when "add_block" then Growth::AddBlock.apply(loaded[:model])
                when "widen_ffn" then Growth::WidenFfn.apply(loaded[:model], layer: @options.fetch(:layer, 0), size: required(:size))
                else raise ArgumentError, "operation must be add_block or widen_ffn"
                end
        optimizer = Optim::AdamW.new(model.named_parameters, learning_rate: model.config[:training]["learning_rate"])
        optimizer.load_state_dict(loaded[:metadata]["optimizer"], allow_growth: true) if loaded[:metadata]["optimizer"]
        state = loaded[:metadata]["training"].merge("parent" => loaded[:path], "growth_operation" => @options[:operation],
          "architecture_version" => loaded[:metadata]["training"].fetch("architecture_version", 1) + 1)
        state["growth"] = nil
        state["growth_warmup_until"] = state.fetch("step", 0) + model.config[:training]["eval_every"]
        { "checkpoint" => Checkpoint.save(required(:output), model: model, tokenizer: loaded[:tokenizer], optimizer: optimizer, training_state: state),
         "parameters" => model.parameter_count, "calibration_invalidated" => true }
      end

      def command_inspect
        loaded = Checkpoint.load(required(:checkpoint))
        { "path" => loaded[:path], "parameters" => loaded[:model].parameter_count,
         "config" => loaded[:model].config.to_h, "tokenizer_fingerprint" => loaded[:tokenizer].fingerprint,
         "step" => loaded[:metadata]["training"]["step"], "calibrated" => !loaded[:metadata]["calibration"].nil? }
      end

      private

      def train_task(task)
        raise ArgumentError, "Use either --resume or --init" if @options[:resume] && @options[:init]
        dataset = Data::Dataset.new(required(:data), kind: task)
        validation = @options[:validation] && Data::Dataset.new(@options[:validation], kind: task)
        output = required(:output)
        if @options[:resume]
          loaded = Checkpoint.load(@options[:resume])
          raise ArgumentError, "Resume task differs from command" unless loaded[:metadata]["training"]["task"] == task.to_s
          raise ArgumentError, "Inference-only checkpoint cannot resume optimizer; use --init" unless loaded[:metadata]["optimizer"]
          raise ArgumentError, "--config and --tokenizer are not accepted with resume; only --steps and --device may change" if @options[:config] || @options[:tokenizer]
          trainer = Trainer.resume(@options[:resume], dataset: dataset, validation: validation, output: output,
            steps: @options[:steps], device: @options[:device])
        else
          model, tokenizer, restored = initial_model(task)
          trainer = Trainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: output,
            validation: validation, task: task, restored: restored, device: @options[:device])
        end
        progress = Progress.new(task: task, total: trainer.model.config[:training]["steps"], out: @err) if @options[:progress]
        path = trainer.train { |state, loss| progress&.update(state, loss, device: trainer.device) }
        { "checkpoint" => path, "step" => trainer.state["step"], "device" => trainer.device,
         "best_checkpoint" => trainer.state["best_checkpoint"], "best_step" => trainer.state["best_step"],
         "stop_reason" => trainer.state["stop_reason"],
         "parameters" => trainer.model.parameter_count, "last_train_loss" => trainer.state["last_train_loss"] }
      end

      def initial_model(task)
        if @options[:init]
          raise ArgumentError, "--config and --tokenizer cannot override an initialized model" if @options[:config] || @options[:tokenizer]
          loaded = Checkpoint.load(@options[:init])
          config = loaded[:model].config
          config = config.with(training: { steps: @options[:steps] }) if @options[:steps]
          model = ChoiceModel.new(config)
          model.load_state_dict(loaded[:model].state_dict)
          state = { "step" => 0, "examples_seen" => 0, "architecture_version" => loaded[:metadata]["training"].fetch("architecture_version", 1),
            "history" => [], "task" => task.to_s, "datasets" => {}, "groups" => loaded[:metadata]["training"].fetch("groups", {}),
            "parent" => loaded[:path] }
          state["groups"] = { "train" => [], "validation" => [] }.merge(state["groups"])
          return [model, loaded[:tokenizer], { "training" => state }]
        end
        config = @options[:config] ? Config.load(@options[:config]) : Config.new
        config = config.with(training: { steps: @options[:steps] }) if @options[:steps]
        tokenizer = Tokenizers::Registry.load(required(:tokenizer))
        Torch.manual_seed(config[:training]["seed"])
        [ChoiceModel.new(config), tokenizer, nil]
      end

      def required(key)
        @options.fetch(key) { raise ArgumentError, "--#{key.to_s.tr('_', '-')} is required" }
      end

      def option_parser(command)
        OptionParser.new do |parser|
          parser.banner = "Usage: bin/easy-ai #{command} [options]"
          if command == "pipeline"
            parser.separator "Defaults: small model; 5 languages, 200 rows per locale/split, 8 choices; MLM 100 steps, choice 300 steps."
            parser.separator "Pipeline --data is a prepared directory. --output must be a new directory. Charts require gnuplot."
          end
          %i[output archive data input config tokenizer validation checkpoint resume init device backend sha256 descriptions language reference_data].each do |key|
            parser.on("--#{key.to_s.tr('_', '-')} VALUE") { |value| @options[key] = value }
          end
          %i[candidates limit train_limit seed vocab_size steps layer size mlm_steps choice_steps eval_every sanity_families].each do |key|
            parser.on("--#{key.to_s.tr('_', '-')} N", Integer) { |value| @options[key] = value }
          end
          parser.on("--locales LIST", Array) { |value| @options[:locales] = value }
          parser.on("--operation NAME") { |value| @options[:operation] = value.tr("-", "_") }
          parser.on("--progress", "Print training progress to stderr") { @options[:progress] = true }
          parser.on("--full-data", "Pipeline: use all selected MASSIVE rows") { @options[:full_data] = true }
          parser.on("--semantic-diagnostics", "Diagnose state and question dependence by semantic task") { @options[:semantic_diagnostics] = true }
          parser.on("--controls", "Relation evaluation: also remove state/question separately") { @options[:controls] = true }
          parser.on("--batch-size N", Integer, "Relation evaluation batch size (default 16)") { |value| @options[:batch_size] = value }
          parser.on("-h", "--help") { @out.puts(parser); @options[:help] = true }
        end
      end

      def help
        @out.puts("Usage: bin/easy-ai COMMAND [options]\nCommands: #{COMMANDS.join(', ')}\nUse COMMAND --help for flags. See docs/decision/usage.md.")
        0
      end
    end
  end
end
