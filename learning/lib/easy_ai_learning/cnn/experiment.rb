module EasyAILearning
  module CNN
    module Experiment
      module_function

      def run(c)
        rows, labels = Course::Data.images(seed: c.seed)
        valid, vlabels = Course::Data.images(seed: c.seed + 1)
        c.artifacts.json("data", { train: [rows, labels], validation: [valid, vlabels] })
        x, y, vx, vy = c.tensor(rows), c.tensor(labels, integer: true), c.tensor(valid), c.tensor(vlabels, integer: true)
        { mlp: BasicNN::Mlp.new(input: 64, hidden: 12), cnn: Model.new, batchnorm: Model.new(normalize: true) }.each do |name, raw|
          model = c.model(raw)
          tx, tv = name == :mlp ? [x.reshape([64, 64]), vx.reshape([64, 64])] : [x, vx]
          c.train(name, model, validation: -> { Training::Math.masked_cross_entropy(model.call(tv), vy) }) { Training::Math.masked_cross_entropy(model.call(tx), y) }
          c.results[name] = c.evaluate(model) do
            logits = model.call(tv)
            { accuracy: c.accuracy(logits, vy), confusion: Foundations::Math.confusion(logits.argmax(-1).cpu.to_a, vlabels),
              parameters: model.parameters.sum(&:numel) }
          end
          c.save("#{name}-model", model, config: name == :mlp ? { input: 64, hidden: 12, output: 2 } : { classes: 2, channels: 4, normalize: name == :batchnorm })
        end
        ae = c.model(Autoencoder::Convolutional.new)
        c.train("conv-ae", ae, validation: -> { Torch::NN::Functional.mse_loss(ae.call(vx), vx) }) { Torch::NN::Functional.mse_loss(ae.call(x), x) }
        c.results[:conv_autoencoder] = c.evaluate(ae) { { validation_mse: Torch::NN::Functional.mse_loss(ae.call(vx), vx).item } }
        c.artifacts.json("reconstruction", c.evaluate(ae) { ae.call(vx).cpu.to_a })
        c.artifacts.image_grid("image-reconstruction", { input: valid, reconstruction: c.evaluate(ae) { ae.call(vx).cpu.to_a } })
        c.save("conv-ae-model", ae)
      end
    end
  end
end
