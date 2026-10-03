module EasyAILearning
  module Course
    class Context
      attr_reader :seed, :steps, :device, :artifacts, :options, :results

      def initialize(options)
        @options = options
        @seed, @steps = options.fetch(:seed), options.fetch(:steps)
        raise ArgumentError, "Positive steps required" unless steps > 0
        @device = EasyAI::Runtime::DevicePolicy.new(requested: options.fetch(:device)).resolve
        @artifacts = Artifacts.new(options.fetch(:output))
        @results = {}
        Torch.manual_seed(seed)
      end

      def tensor(values, integer: false)
        Data.tensor(values, device: device, integer: integer)
      end

      def model(model)
        model.to(device)
        model
      end

      def train(name, model, validation: nil, stopping: nil, **settings, &objective)
        loop = Training::Loop.new(model, seed: seed, **settings).run(steps: steps, validation: validation, stopping: stopping, &objective)
        artifacts.json("#{name}-history", loop.history)
        series = { training: loop.history.map { |r| [r[:step], r[:train_loss]] } }
        series[:validation] = loop.history.map { |r| [r[:step], r[:validation_loss]] } if validation
        artifacts.plot("#{name}-loss", series, title: "#{name}: loss (linear scale)")
        artifacts.json("#{name}-diagnostics", Diagnostics::Stats.model(model))
        artifacts.json("#{name}-optimizer", loop.optimizer.state_dict)
        artifacts.json("#{name}-training-state", loop.state_dict)
        loop
      end

      def evaluate(model)
        previous = model.training
        model.eval
        Torch.no_grad { yield }
      ensure
        model.train(previous)
      end

      def accuracy(logits, labels)
        logits.argmax(-1).eq(labels).to(dtype: :float32).mean.item
      end

      def save(name, model, config: {})
        artifacts.save_model(name, model, config: config)
      end

      def finish(chapter)
        report = { chapter: chapter, seed: seed, steps: steps, device: device.to_s, ruby: RUBY_VERSION, torch_rb: Gem.loaded_specs.fetch("torch-rb").version.to_s,
          recorded_at: Time.now.utc.iso8601,
          results: results, note: "Observed educational run; no convergence or generalization guarantee." }
        artifacts.json("results", report)
        puts JSON.pretty_generate(report)
        report
      end
    end
  end
end
