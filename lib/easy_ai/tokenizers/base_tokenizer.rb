require "logger"
require "json"
require "active_support/logger"
require "active_support/isolated_execution_state"

module EasyAI
  module Tokenizers
    class BaseTokenizer
      UNKNOWN_TOKEN = "<|unk|>".freeze

      attr_reader :logger, :token_to_id, :id_to_token

      def initialize(logger: EasyAI.logger)
        @logger = logger
        @token_to_id = { UNKNOWN_TOKEN => 0 }
        @id_to_token = { 0 => UNKNOWN_TOKEN }
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
        tokenize(text).map { |token| lookup_token_id(token) }
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
          mapping = { UNKNOWN_TOKEN => 0 }
          tokens.flatten.each do |token|
            key = token.to_s
            next if mapping.key?(key)

            mapping[key] = mapping.length
          end
          assign_token_mappings!(mapping)
        end

        def assign_token_mappings!(mapping)
          normalized = mapping.transform_keys(&:to_s)
          normalized[UNKNOWN_TOKEN] ||= 0
          normalized = normalized.sort_by { |token, id| token == UNKNOWN_TOKEN ? [-1, 0] : [0, id] }.each_with_index.to_h do |(token, _), idx|
            [token, idx]
          end

          self.token_to_id = normalized
          self.id_to_token = token_to_id.invert
        end

        def lookup_token_id(token)
          key = token.to_s
          token_to_id.fetch(key, unknown_token_id)
        end

        def unknown_token_id
          token_to_id[UNKNOWN_TOKEN] || 0
        end

      private
    end
  end
end
