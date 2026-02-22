require "tokenizers"

module EasyAI
  module Tokenizers
    class QwenBpe < BaseTokenizer
      attr_reader :tokenizer, :model_name

      # Pre-trained Qwen models available
      PRETRAINED_MODELS = {
        "qwen3-0.6b" => "Qwen/Qwen3-0.6B",
        "qwen3-1.7b" => "Qwen/Qwen3-1.7B",
        "qwen3-4b" => "Qwen/Qwen3-4B",
        "qwen3-8b" => "Qwen/Qwen3-8B",
        "qwen3-14b" => "Qwen/Qwen3-14B",
        "qwen3-32b" => "Qwen/Qwen3-32B",
        "qwen3-30b-a3b" => "Qwen/Qwen3-30B-A3B",
        "qwen3-next-80b-a3b" => "Qwen/Qwen3-Next-80B-A3B-Instruct"
      }.freeze

      DEFAULT_MODEL = "qwen3-next-80b-a3b".freeze

      def initialize(model_name: nil, **kwargs)
        super(**kwargs)
        @model_name = model_name.presence || DEFAULT_MODEL
        @tokenizer = load_tokenizer(model_name)
      end

      def train(text)
        # Qwen tokenizer is pre-trained and cannot be retrained
        logger.warn { "#{self.class}: Tokenizer is pre-trained, training is not supported. Skipping." }
      end

      def tokenize(text)
        logger.info { "#{self.class} tokenize text: #{EasyAI::Logger.summary(text)}" }

        # Use huggingface tokenizers gem to encode
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens

        logger.info { "#{self.class} tokenize result: #{tokens.length} tokens" }
        tokens
      end

      def encode(text)
        encoding = tokenizer.encode(text)
        encoding.ids
      end

      def decode(ids)
        tokenizer.decode(ids)
      end

      def detokenize(tokens)
        # For tokenizers gem, we need to decode ids, not tokens
        # If tokens are strings, we need to convert them back to ids first
        # This is a simplified version - in practice you'd track ids
        logger.info { "#{self.class} detokenize tokens: #{tokens.length} tokens" }

        # Join tokens if they're strings (fallback)
        if tokens.all? { |t| t.is_a?(String) }
          result = tokens.join
          logger.info { "#{self.class} detokenize result: #{result.truncate(100)}" }
          return result
        end

        # If tokens are IDs, decode them
        tokenizer.decode(tokens)
      end

      def vocab_size
        tokenizer.vocab_size
      end

      private

        def load_tokenizer(name)
          model_id = PRETRAINED_MODELS[name] || name
          logger.info { "#{self.class} loading pre-trained tokenizer: #{model_id}" }

          begin
            # Load from Hugging Face Hub using top-level Tokenizers module
            ::Tokenizers.from_pretrained(model_id)
          rescue => e
            logger.error { "Failed to load tokenizer #{model_id}: #{e.message}" }
            raise
          end
        end
    end
  end
end
