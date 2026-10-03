module EasyAILearning
  module Seq2seq
    module Experiment
      module_function

      def run(c)
        rows = Course::Data.reversal(seed: c.seed)
        valid = Course::Data.reversal(seed: c.seed + 1)
        c.artifacts.json("data", { train: rows, validation: valid })
        x, decoder, y = rows.map { |v| c.tensor(v, integer: true) }
        vx, vd, vy = valid.map { |v| c.tensor(v, integer: true) }
        model = c.model(Model.new)
        c.train("seq2seq", model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx, vd), vy) }) do
          Training::Math.masked_cross_entropy(model.call(x, decoder), y)
        end
        c.results[:seq2seq] = c.evaluate(model) do
          generated = model.generate(vx)
          { teacher_forced_accuracy: c.accuracy(model.call(vx, vd), vy), free_token_accuracy: generated.eq(vy).to(dtype: :float32).mean.item,
            exact_sequence_accuracy: generated.eq(vy).to(dtype: :int64).sum(1).eq(5).to(dtype: :float32).mean.item }
        end
        c.artifacts.json("predictions", { source: valid.first, target: valid.last, generated: model.generate(vx).cpu.to_a })
        c.save("seq2seq-model", model, config: { vocab: 7, hidden: 12 })
      end
    end
  end
end
