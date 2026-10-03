require "json"

module EasyAILearning
  module Tokenizers
    # Integer byte symbols preserve every UTF-8 byte, including whitespace.
    # Unlike the historical word-splitting examples, unseen text needs no UNK.
    class ReversibleBpe
      attr_reader :merges

      def initialize(num_merges: 16)
        raise ArgumentError, "Nonnegative merge count required" unless num_merges >= 0
        @limit, @merges = num_merges, []
      end

      def train(text)
        @merges = []
        symbols = text.encode("UTF-8").bytes
        @limit.times do
          counts = Hash.new(0)
          symbols.each_cons(2) { |pair| counts[pair] += 1 }
          pair, count = counts.max_by { |p, n| [n, -p[0], -p[1]] }
          break unless pair && count >= 2
          replacement = 256 + merges.size
          merges << [pair, replacement]
          symbols = apply(symbols, pair, replacement)
        end
        self
      end

      def encode(text)
        merges.reduce(text.encode("UTF-8").bytes) { |tokens, (pair, id)| apply(tokens, pair, id) }
      end

      def decode(ids)
        pieces = Array.new(256) { |i| [i] }
        merges.each { |(a, b), id| pieces[id] = pieces.fetch(a) + pieces.fetch(b) }
        ids.flat_map { |id| pieces.fetch(id) }.pack("C*").force_encoding("UTF-8")
      end

      def vocab_size
        256 + merges.size
      end

      def save(path)
        File.write(path, JSON.pretty_generate(format: 1, merges: merges))
      end

      def self.load(path)
        saved = JSON.parse(File.read(path))
        raise ArgumentError, "Unknown tokenizer format" unless saved.fetch("format") == 1
        model = new(num_merges: saved.fetch("merges").size)
        model.instance_variable_set(:@merges, saved.fetch("merges"))
        model
      end

      private

      def apply(tokens, pair, replacement)
        result, index = [], 0
        while index < tokens.size
          if tokens[index, 2] == pair
            result << replacement
            index += 2
          else
            result << tokens[index]
            index += 1
          end
        end
        result
      end
    end
  end
end
