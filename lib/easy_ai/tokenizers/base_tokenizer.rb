require "logger"
require "json"
require "active_support/logger"
require "active_support/isolated_execution_state"

module EasyAI
  module Tokenizers
    class BaseTokenizer
      attr_reader :logger, :token_to_id, :id_to_token

      def initialize(logger: default_logger)
        @logger = logger
        @token_to_id = {}
        @id_to_token = {}
      end

      def train(_corpus)
        raise NotImplementedError
      end

      def tokenize(_text)
        raise NotImplementedError
      end

      def detokenize(_tokens)
        raise NotImplementedError
      end

      def encode(text)
        tokenize(text).map { |token| token_to_id.fetch(token.to_s) }
      end

      def decode(ids)
        tokens = ids.map { |id| id_to_token.fetch(id) }
        detokenize(tokens)
      end

      def save_vocab(path)
        serialized = { token_to_id: token_to_id }
        File.write(path, JSON.pretty_generate(serialized))
      end

      def load_vocab(path)
        serialized = JSON.parse(File.read(path))
        assign_token_mappings!(serialized.fetch("token_to_id"))
      end

      def vocab_size
        token_to_id.size
      end

      protected

        attr_writer :token_to_id, :id_to_token

        def rebuild_token_mappings!(tokens)
          unique_tokens = tokens.flatten.uniq
          mapping = unique_tokens.each_with_index.to_h { |token, index| [token.to_s, index] }
          assign_token_mappings!(mapping)
        end

        def assign_token_mappings!(mapping)
          self.token_to_id = mapping.transform_keys(&:to_s)
          self.id_to_token = token_to_id.invert
        end

      private

        def default_logger
          ActiveSupport::Logger.new(
            ENV.fetch("LOG_PATH", STDOUT),
            level: ENV.fetch("LOG_LEVEL", "INFO")
          )
        end
    end
  end
end
