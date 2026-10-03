module EasyAILearning
  module Training
    module Experiment
      module_function

      def run(c)
        rows, labels = Course::Data.classification(seed: c.seed)
        valid, vlabels = Course::Data.classification(seed: c.seed + 1)
        c.artifacts.json("data", { train: [rows, labels], validation: [valid, vlabels] })
        stats = Math.standardize_fit(rows)
        x, y = c.tensor(Math.standardize(rows, stats)), c.tensor(labels, integer: true)
        vx, vy = c.tensor(Math.standardize(valid, stats)), c.tensor(vlabels, integer: true)
        c.artifacts.json("standardization", stats)
        variants = { sgd: [:sgd, 0.1, 0, 0], momentum: [:momentum, 0.1, 0, 0], adam: [:adam, 0.02, 0, 0],
          adamw: [:adamw, 0.02, 0, 0], decay: [:adamw, 0.02, 0.1, 0], dropout: [:adamw, 0.02, 0, 0.2] }
        variants.each do |name, (kind, lr, decay, dropout)|
          Torch.manual_seed(c.seed)
          model = c.model(BasicNN::Mlp.new(dropout: dropout))
          validation = -> { Math.masked_cross_entropy(model.call(vx), vy) }
          loop = c.train(name, model, kind: kind, lr: lr, decay: decay, validation: validation) { Math.masked_cross_entropy(model.call(x), y) }
          c.results[name] = c.evaluate(model) do
            { accuracy: c.accuracy(model.call(vx), vy), train_loss: loop.history.last[:train_loss], validation_loss: validation.call.item }
          end
          c.save("#{name}-model", model, config: { input: 2, hidden: 12, output: 2, dropout: dropout })
        end
        stopped = c.model(BasicNN::Mlp.new)
        stopping = EarlyStopping.new(patience: 5, minimum_delta: 0.001)
        loop = c.train("early-stopping", stopped, stopping: stopping,
          validation: -> { Math.masked_cross_entropy(stopped.call(vx), vy) }) { Math.masked_cross_entropy(stopped.call(x), y) }
        c.results[:early_stopping] = { steps: loop.updates, best_validation_loss: stopping.best }
        c.save("early-stopping-model", stopped, config: { input: 2, hidden: 12, output: 2 })
        control = Torch::NN::Parameter.new(c.tensor([1.0]))
        Accumulation.backward([-> { (control * c.tensor([1.0])).square.mean }, -> { (control * c.tensor([2.0, 3.0])).square.mean }], counts: [1, 2])
        c.results[:accumulated_gradient] = control.grad.item
        control.grad.zero!
        scaler = GradScaler.new
        scaler.backward(control.square.sum)
        optimizer = Optimizer.new({ "scalar" => control }, kind: :sgd, lr: 0.1)
        scaler.step(optimizer)
        c.results[:loss_scaling] = { value_after_sgd: control.item, scale: scaler.scale,
          explicit_half_roundtrip: c.tensor([0.1]).to(dtype: :float16).to(dtype: :float32).cpu.to_a,
          note: "Explicit loss scaling and half rounding; no automatic mixed-precision context is provided by this lesson." }
        scalar = ScalarOptimizer.new(kind: :adamw, lr: 0.1, decay: 0.2)
        c.results[:adamw_first_update] = scalar.update(2.0, 0.5)
        c.artifacts.json("schedule", Array.new(c.steps) { |i| Math.learning_rate(i, total: c.steps, base: 0.02, warmup: c.steps / 10) })
      end
    end
  end
end
