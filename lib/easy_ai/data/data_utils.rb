module EasyAI
  module Data
    module DataUtils
      module_function

      def ensure_tokenizer_trained!(tokenizer, corpus)
        return unless tokenizer.vocab_size <= 1

        tokenizer.train(corpus)
      end

      def deterministic_rand(seed)
        Random.new(seed)
      end
    end
  end
end
