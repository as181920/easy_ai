require "json"

module EasyAILearning
  module Tokenizers
    class Character
      attr_reader :characters

      def train(text)
        @characters = text.chars.uniq.sort
        self
      end

      def encode(text)
        raise ArgumentError, "Train first" unless characters
        text.chars.map { |char| (characters.index(char) || -1) + 1 }
      end

      def decode(ids)
        ids.map { |id| id.zero? ? "�" : characters.fetch(id - 1) }.join
      end

      def vocab_size
        characters.size + 1
      end

      def save(path)
        File.write(path, JSON.pretty_generate(characters: characters))
      end

      def self.load(path)
        new.tap { |model| model.instance_variable_set(:@characters, JSON.parse(File.read(path)).fetch("characters")) }
      end
    end
  end
end
