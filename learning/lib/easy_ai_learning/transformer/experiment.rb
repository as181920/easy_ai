module EasyAILearning
  module Transformer
    module Experiment
      module_function

      def run(c)
        rows, valid = Course::Data.reversal(seed: c.seed), Course::Data.reversal(seed: c.seed + 1)
        c.artifacts.json("data", { train: rows, validation: valid })
        x, decoder, y = rows.map { |v| c.tensor(v, integer: true) }
        vx, vd, vy = valid.map { |v| c.tensor(v, integer: true) }
        # Bidirectional encoder predicts the reversed fixed-length input positions.
        target, vtarget = y.narrow(1, 0, 4), vy.narrow(1, 0, 4)
        variants = { baseline: {}, no_position: { positions: false }, no_residual: { residual: false }, no_norm: { normalize: false }, post_norm: { post_norm: true } }
        variants.each do |name, settings|
          Torch.manual_seed(c.seed)
          model = c.model(SequenceModel.new(**settings))
          c.train(name, model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx), vtarget) }) do
            Training::Math.masked_cross_entropy(model.call(x), target)
          end
          c.results[name] = c.evaluate(model) { { token_accuracy: c.accuracy(model.call(vx), vtarget) } }
          c.save("#{name}-model", model, config: settings)
        end
        model = c.model(EncoderDecoder.new)
        c.train("encoder-decoder", model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx, vd), vy) }) do
          Training::Math.masked_cross_entropy(model.call(x, decoder), y)
        end
        c.results[:encoder_decoder] = c.evaluate(model) { { teacher_accuracy: c.accuracy(model.call(vx, vd), vy), free_accuracy: model.generate(vx).eq(vy).to(dtype: :float32).mean.item } }
        c.save("encoder-decoder-model", model)
        c.artifacts.json("sinusoidal-positions", Sinusoidal.values(8, 8))
      end
    end
  end
end
