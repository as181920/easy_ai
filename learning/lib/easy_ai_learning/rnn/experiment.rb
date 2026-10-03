module EasyAILearning
  module RNN
    module Experiment
      module_function

      def run(c)
        rows, labels = Course::Data.sequences(seed: c.seed)
        valid, vlabels = Course::Data.sequences(seed: c.seed + 1)
        c.artifacts.json("data", { train: [rows, labels], validation: [valid, vlabels] })
        x, y, vx, vy = [rows, labels, valid, vlabels].map { |v| c.tensor(v, integer: true) }
        %i[rnn lstm gru].each do |kind|
          Torch.manual_seed(c.seed)
          model = c.model(Model.new(kind: kind))
          c.train(kind, model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx), vy) }) { Training::Math.masked_cross_entropy(model.call(x), y) }
          c.results[kind] = c.evaluate(model) { { token_accuracy: c.accuracy(model.call(vx), vy) } }
          c.save("#{kind}-model", model, config: { vocab: 6, hidden: 8, kind: kind })
        end
        memory_x, memory_y = Course::Data.delayed_copy(seed: c.seed)
        memory_vx, memory_vy = Course::Data.delayed_copy(seed: c.seed + 1)
        long_x, long_y = Course::Data.delayed_copy(seed: c.seed + 2, length: 20)
        mx, my, mvx, mvy, lx, ly = [memory_x, memory_y, memory_vx, memory_vy, long_x, long_y].map { |v| c.tensor(v, integer: true) }
        c.artifacts.json("memory-data", { train: [memory_x, memory_y], validation: [memory_vx, memory_vy], length20: [long_x, long_y] })
        %i[rnn lstm gru].each do |kind|
          Torch.manual_seed(c.seed)
          model = c.model(Model.new(vocab: 7, kind: kind))
          final_logits = ->(inputs) { output = model.call(inputs); output.narrow(1, output.shape[1] - 1, 1).squeeze(1) }
          c.train("#{kind}-memory", model, validation: -> { Training::Math.masked_cross_entropy(final_logits.call(mvx), mvy) }) do
            Training::Math.masked_cross_entropy(final_logits.call(mx), my)
          end
          c.results["#{kind}_memory"] = c.evaluate(model) do
            { length10_accuracy: c.accuracy(final_logits.call(mvx), mvy), length20_accuracy: c.accuracy(final_logits.call(lx), ly) }
          end
          c.save("#{kind}-memory-model", model, config: { vocab: 7, hidden: 8, kind: kind })
        end
      end
    end
  end
end
