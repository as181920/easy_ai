module EasyAILearning
  module Resnet
    module Experiment
      module_function

      def run(c)
        rows, labels = Course::Data.images(seed: c.seed)
        valid, vlabels = Course::Data.images(seed: c.seed + 1)
        c.artifacts.json("data", { train: [rows, labels], validation: [valid, vlabels] })
        x, y, vx, vy = c.tensor(rows), c.tensor(labels, integer: true), c.tensor(valid), c.tensor(vlabels, integer: true)
        { plain: [false, true], residual: [true, true], without_norm: [true, false] }.each do |name, (residual, normalize)|
          Torch.manual_seed(c.seed)
          config = { residual: residual, normalize: normalize, depth: 3, channels: 4 }
          model = c.model(Model.new(**config))
          c.train(name, model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx), vy) }) { Training::Math.masked_cross_entropy(model.call(x), y) }
          c.results[name] = c.evaluate(model) { { accuracy: c.accuracy(model.call(vx), vy), parameters: model.parameters.sum(&:numel) } }
          c.save("#{name}-model", model, config: config)
        end
        projection = c.model(Block.new(input: 4, output: 8, stride: 2))
        c.results[:bottleneck_shape] = c.model(Bottleneck.new).call(Torch.ones([2, 4, 8, 8], device: c.device)).shape
        c.results[:preactivation_shape] = c.model(Preactivation.new).call(Torch.ones([2, 4, 8, 8], device: c.device)).shape
        c.results[:projection_shape] = projection.call(Torch.ones([2, 4, 8, 8], device: c.device)).shape
      end
    end
  end
end
