require "json"
require "digest"
require "set"

module EasyAI
  module Tokenizers
    # Byte-level BPE implemented in Ruby. The heap/index updates only affected
    # chunks during training; all 256 bytes remain available after training.
    class ByteBpe
      SPECIALS = %w[pad unk cls sep mask eos].freeze
      BYTE_OFFSET = SPECIALS.length
      BASE_SIZE = BYTE_OFFSET + 256
      attr_reader :merges

      def initialize
        @merges = []
        rebuild!
      end

      def id(name)
        SPECIALS.index(name.to_s) || raise(ArgumentError, "Unknown special token #{name}")
      end

      def vocab_size
        BASE_SIZE + merges.length
      end

      def train(texts, vocab_size: 32_000, min_frequency: 2)
        raise ArgumentError, "vocab_size must be >= #{BASE_SIZE}" unless vocab_size >= BASE_SIZE
        raise ArgumentError, "min_frequency must be positive" unless min_frequency >= 1
        texts = [texts] if texts.is_a?(String)
        frequencies = Hash.new(0)
        texts.each { |text| chunks(text).each { |chunk| frequencies[chunk.bytes.map { |b| b + BYTE_OFFSET }] += 1 } }
        raise ArgumentError, "No tokenizer training text" if frequencies.empty?
        words = frequencies.keys
        counts = frequencies.values
        pair_counts = Hash.new(0)
        locations = Hash.new { |h, k| h[k] = Set.new }
        words.each_with_index do |word, index|
          word.each_cons(2) do |pair|
            pair_counts[pair] += counts[index]
            locations[pair] << index
          end
        end
        heap = []
        pair_counts.each { |pair, count| heap_push(heap, [-count, *pair]) }
        @merges = []
        while self.vocab_size < vocab_size && !heap.empty?
          negative, left, right = heap_pop(heap)
          pair = [left, right]
          next unless pair_counts[pair] == -negative
          break if -negative < min_frequency
          replacement = self.vocab_size
          affected = locations[pair].to_a
          changed = Set.new
          affected.each do |index|
            word = words[index]
            word.each_cons(2) do |old_pair|
              pair_counts[old_pair] -= counts[index]
              locations[old_pair].delete(index)
              changed << old_pair
            end
            words[index] = merge_pair(word, pair, replacement)
            words[index].each_cons(2) do |new_pair|
              pair_counts[new_pair] += counts[index]
              locations[new_pair] << index
              changed << new_pair
            end
          end
          @merges << pair
          changed.each { |p| heap_push(heap, [-pair_counts[p], *p]) if pair_counts[p] > 0 }
        end
        rebuild!
        self
      end

      def encode(text)
        chunks(text).flat_map do |chunk|
          tokens = chunk.bytes.map { |b| b + BYTE_OFFSET }
          loop do
            pair = tokens.each_cons(2).select { |p| @ranks.key?(p) }.min_by { |p| @ranks.fetch(p) }
            break unless pair
            tokens = merge_pair(tokens, pair, BASE_SIZE + @ranks.fetch(pair))
          end
          tokens
        end
      end

      def decode(ids, skip_special: true)
        bytes = ids.flat_map do |token|
          raise ArgumentError, "Invalid token id #{token.inspect}" unless token.is_a?(Integer) && token.between?(0, vocab_size - 1)
          if token < BYTE_OFFSET
            raise ArgumentError, "Special tokens have no text representation" unless skip_special
            []
          else
            @pieces.fetch(token)
          end
        end
        bytes.pack("C*").force_encoding(Encoding::UTF_8)
      end

      # Byte intervals in the original text, including byte fragments of CJK characters.
      def encode_with_offsets(text)
        position = 0
        ids = encode(text)
        offsets = ids.map do |token|
          start = position
          position += @pieces.fetch(token).size
          [start, position]
        end
        [ids, offsets]
      end

      def to_h
        { "format" => "easy_ai.byte_bpe", "version" => 1, "specials" => SPECIALS, "merges" => merges }
      end

      def fingerprint
        Digest::SHA256.hexdigest(JSON.generate(to_h))
      end

      def save(path)
        File.write(path, JSON.pretty_generate(to_h))
      end

      def self.load(path)
        from_h(JSON.parse(File.read(path)))
      end

      def self.from_h(data)
        raise ArgumentError, "Unsupported tokenizer format" unless data.values_at("format", "version", "specials") == ["easy_ai.byte_bpe", 1, SPECIALS]
        tokenizer = new
        seen = Set.new
        data.fetch("merges").each_with_index do |pair, rank|
          valid = pair.is_a?(Array) && pair.length == 2 && pair.all? { |i| i.is_a?(Integer) && i.between?(BYTE_OFFSET, BASE_SIZE + rank - 1) }
          raise ArgumentError, "Invalid/duplicate BPE merge" unless valid && seen.add?(pair)
        end
        tokenizer.instance_variable_set(:@merges, data.fetch("merges").map(&:dup))
        tokenizer.send(:rebuild!)
        tokenizer
      end

      private

      def chunks(text)
        raise ArgumentError, "Expected valid UTF-8 text" unless text.is_a?(String) && text.encoding.ascii_compatible? && text.dup.force_encoding("UTF-8").valid_encoding?
        # Chunking is identical at training/inference; whitespace is never dropped.
        text.encode("UTF-8").scan(/\p{L}{1,64}|\p{N}{1,64}|[^\p{L}\p{N}\s]{1,64}|\s{1,64}/)
      end

      def rebuild!
        @ranks = merges.each_with_index.to_h
        @pieces = (0...BYTE_OFFSET).map { [] } + (0..255).map { |b| [b] }
        merges.each { |left, right| @pieces << @pieces.fetch(left) + @pieces.fetch(right) }
      end

      def merge_pair(tokens, pair, replacement)
        result = []
        i = 0
        while i < tokens.length
          if tokens[i] == pair[0] && tokens[i + 1] == pair[1]
            result << replacement
            i += 2
          else
            result << tokens[i]
            i += 1
          end
        end
        result
      end

      def heap_push(heap, value)
        heap << value
        i = heap.length - 1
        while i > 0
          parent = (i - 1) / 2
          break if (heap[parent] <=> value) <= 0
          heap[i] = heap[parent]
          i = parent
        end
        heap[i] = value
      end

      def heap_pop(heap)
        first = heap.first
        last = heap.pop
        unless heap.empty?
          i = 0
          while (child = 2 * i + 1) < heap.length
            child += 1 if child + 1 < heap.length && (heap[child + 1] <=> heap[child]) < 0
            break if (last <=> heap[child]) <= 0
            heap[i] = heap[child]
            i = child
          end
          heap[i] = last
        end
        first
      end
    end
  end
end
