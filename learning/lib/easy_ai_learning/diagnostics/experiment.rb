module EasyAILearning
  module Diagnostics
    module Experiment
      module_function

      def run(c)
        rows, labels = Course::Data.classification(seed: c.seed)
        valid, vlabels = Course::Data.classification(seed: c.seed + 1)
        c.artifacts.json("data", { train: [rows, labels], validation: [valid, vlabels] })
        x, y, vx, vy = c.tensor(rows), c.tensor(labels, integer: true), c.tensor(valid), c.tensor(vlabels, integer: true)
        { baseline: [0.02, 0.0], excessive_decay: [0.02, 8.0], small_lr: [1e-6, 0.0] }.each do |name, (lr, decay)|
          Torch.manual_seed(c.seed)
          model = c.model(BasicNN::Mlp.new)
          before = Stats.snapshot(model)
          c.artifacts.json("#{name}-initial", Stats.model(model))
          c.train(name, model, lr: lr, decay: decay, validation: -> { Training::Math.masked_cross_entropy(model.call(vx), vy) }) do
            Training::Math.masked_cross_entropy(model.call(x), y)
          end
          c.results[name] = c.evaluate(model) do
            { validation_accuracy: c.accuracy(model.call(vx), vy), activation: Stats.activation(model.last_hidden), updates: Stats.updates(model, before) }
          end
          histograms = model.named_parameters.to_h { |key, p| [key, Stats.histogram(p.detach.cpu.to_a)] }
          c.artifacts.plot("#{name}-weights", histograms, title: "#{name}: weight histograms by parameter tensor")
          c.results[name][:hidden_spectrum] = Stats.spectrum(model.hidden.weight)
          c.save("#{name}-model", model, config: { input: 2, hidden: 12, output: 2 })
        end
        c.results[:concentrated_but_valid] = Stats.summarize(Array.new(100, 0.0) + [0.01, -0.01])
        c.results[:nonfinite_detection] = Stats.summarize([0, Float::INFINITY, Float::NAN])
        c.results[:diagnosis] = "Compare layers, activations, gradients and held-out performance; no histogram acceptance threshold."
      end
    end
  end
end
