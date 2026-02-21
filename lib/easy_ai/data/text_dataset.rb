module EasyAI
  module Data
    class TextDataset
      attr_reader :tokenizer, :block_size, :token_ids

      def initialize(tokenizer:, text: nil, path: nil, block_size: 128, auto_train: true)
        raise ArgumentError, "Provide text or path" if text.blank? && path.blank?

        @tokenizer = tokenizer
        @block_size = block_size
        corpus = text || File.read(path)
        DataUtils.ensure_tokenizer_trained!(tokenizer, corpus) if auto_train
        @token_ids = tokenizer.encode(corpus)
      end

      def length
        token_ids.length
      end

      def [](range)
        token_ids[range]
      end

      def to_tensor
        Torch.tensor(token_ids, dtype: :int64)
      end
    end
  end
end
