module EasyAILearning
  module Tokenizers
    module Experiment
      module_function

      def run(c)
        training = "hello world\n你好 世界\nhello learning\n你好 learning\n" * 8
        validation = "hello 世界\nnew words! 你好\t🙂"
        c.artifacts.json("data", { train: training, validation: validation })
        { character: Character.new.train(training), byte: ReversibleBpe.new(num_merges: 0).train(training),
          bpe: ReversibleBpe.new(num_merges: 24).train(training) }.each do |name, tokenizer|
          ids = tokenizer.encode(validation)
          decoded = tokenizer.decode(ids)
          c.results[name] = { vocabulary: tokenizer.vocab_size, tokens: ids.size, roundtrip: decoded == validation,
            decoded: decoded, embedding_parameters_for_width_8: tokenizer.vocab_size * 8 }
          tokenizer.save(File.join(c.artifacts.directory, "#{name}-tokenizer.json"))
          c.artifacts.json("#{name}-encoding", ids)
        end
        # Embedding receives gradients only at looked-up rows.
        embedding = c.model(Torch::NN::Embedding.new(8, 3))
        ids = c.tensor([1, 2, 1], integer: true)
        embedding.call(ids).sum.backward
        c.results[:embedding] = { gradients: embedding.weight.grad.cpu.to_a, shape: embedding.call(ids).shape }
        c.save("embedding-model", embedding, config: { num_embeddings: 8, embedding_dim: 3 })
        c.artifacts.plot("token-length", c.results.slice(:character, :byte, :bpe).to_h { |k, v| [k, [[0, v[:tokens]], [1, v[:vocabulary]]]] },
          title: "x=0 token count; x=1 vocabulary size")
      end
    end
  end
end
