module EasyAILearning
  module Attention
    module Experiment
      module_function

      def run(c)
        rows, valid = Course::Data.reversal(seed: c.seed), Course::Data.reversal(seed: c.seed + 1)
        c.artifacts.json("data", { train: rows, validation: valid })
        x, decoder, y = rows.map { |v| c.tensor(v, integer: true) }
        vx, vd, vy = valid.map { |v| c.tensor(v, integer: true) }
        [false, true].each do |enabled|
          Torch.manual_seed(c.seed)
          model = c.model(Seq2seq::Model.new(attention: enabled))
          name = enabled ? "cross-attention" : "fixed-state"
          c.train(name, model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx, vd), vy) }) do
            Training::Math.masked_cross_entropy(model.call(x, decoder), y)
          end
          c.results[name] = c.evaluate(model) { { teacher_accuracy: c.accuracy(model.call(vx, vd), vy),
            free_accuracy: model.generate(vx).eq(vy).to(dtype: :float32).mean.item } }
          if enabled
            weights = model.instance_variable_get(:@attention).last_weights.detach.cpu.to_a
            c.artifacts.json("alignment", weights)
          end
          c.save("#{name}-model", model, config: { vocab: 7, hidden: 12, attention: enabled })
        end
        attention = c.model(MultiHead.new(embed_dim: 4, num_heads: 2))
        input = c.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
        output = attention.call(input, causal: true)
        changed = input.clone
        changed[0][2] = c.tensor([9, 9, 9, 9])
        altered = attention.call(changed, causal: true)
        c.results[:causal_past_difference] = (output.narrow(1, 0, 2) - altered.narrow(1, 0, 2)).abs.max.item
        c.artifacts.json("causal-weights", attention.last_weights.detach.cpu.to_a)
        additive = c.model(Additive.new(hidden: 4))
        c.results[:additive_context] = additive.call(input.narrow(1, 0, 1).squeeze(1), input).detach.cpu.to_a
      end
    end
  end
end
