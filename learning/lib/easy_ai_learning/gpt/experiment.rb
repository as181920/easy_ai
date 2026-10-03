module EasyAILearning
  module GPT
    module Experiment
      module_function

      def run(c)
        train_rows, train_labels = Course::Data.sequences(seed: c.seed, length: 8)
        valid_rows, valid_labels = Course::Data.sequences(seed: c.seed + 1, length: 8)
        test_rows, test_labels = Course::Data.sequences(seed: c.seed + 2, length: 8)
        c.artifacts.json("data", { train: [train_rows, train_labels], validation: [valid_rows, valid_labels], test: [test_rows, test_labels],
          vocabulary: %w[a b c d e f], recipe: "Cyclic six-symbol language, independent start-position samples; not natural language." })
        x, y, vx, vy, tx, ty = [train_rows, train_labels, valid_rows, valid_labels, test_rows, test_labels].map { |v| c.tensor(v, integer: true) }
        config = { vocab_size: 6, block_size: 8, n_layer: 1, n_head: 2, n_embd: 8, dropout: 0.1 }
        model = c.model(Model.new(config))
        c.train("gpt", model, lr: 0.02, decay: 0.01, schedule: true,
          validation: -> { Training::Math.masked_cross_entropy(model.call(vx), vy) }) { Training::Math.masked_cross_entropy(model.call(x), y) }
        c.results[:gpt] = c.evaluate(model) do
          loss = Training::Math.masked_cross_entropy(model.call(tx), ty).item
          { test_loss: loss, test_perplexity: ::Math.exp(loss), test_token_accuracy: c.accuracy(model.call(tx), ty), parameters: model.parameters.sum(&:numel) }
        end
        c.results[:generation] = model.generate(c.tensor([[0]], integer: true), max_new_tokens: 12, temperature: 0.7, top_k: 2).cpu.to_a
        c.save("gpt-model", model, config: config)
      end
    end
  end
end
