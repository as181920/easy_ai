require "json"

module EasyAI
  module Tokenizers
    class VocabStore
      def self.save(path, tokenizer)
        payload = {
          token_to_id: tokenizer.token_to_id,
          merges: serialize_merges(tokenizer)
        }
        File.write(path, JSON.pretty_generate(payload))
      end

      def self.load(path, tokenizer)
        payload = JSON.parse(File.read(path))
        tokenizer.__send__(:assign_token_mappings!, payload.fetch("token_to_id"))
        if payload["merges"] && tokenizer.respond_to?(:merges)
          tokenizer.merges.replace(payload["merges"].each_with_object({}) do |entry, memo|
            memo[entry.fetch("pair").map { |token| token }] = entry.fetch("replacement")
          end)
        end
      end

      def self.serialize_merges(tokenizer)
        return [] unless tokenizer.respond_to?(:merges)

        tokenizer.merges.map do |pair, replacement|
          { pair: pair, replacement: replacement }
        end
      end
      private_class_method :serialize_merges
    end
  end
end
