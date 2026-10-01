require "tokenizers"
require "tempfile"
require "json"
require "digest"

module EasyAI
  module Tokenizers
    # Optional Rust-backed implementation, trained only on the supplied corpus.
    # Its tokenizer/checkpoint format is explicitly distinct from the Ruby BPE.
    class NativeBpe
      SPECIALS = ByteBpe::SPECIALS.map { |name| "[#{name.upcase}]" }.freeze

      def initialize(backend = nil)
        @backend = backend || ::Tokenizers::Tokenizer.new(::Tokenizers::Models::BPE.new(unk_token: "[UNK]"))
        unless backend
          @backend.pre_tokenizer = ::Tokenizers::PreTokenizers::ByteLevel.new(add_prefix_space: false)
          @backend.decoder = ::Tokenizers::Decoders::ByteLevel.new
        end
      end

      def train(texts, vocab_size: 32_000, min_frequency: 2)
        raise ArgumentError, "Vocabulary must include all bytes and special tokens" if vocab_size < ByteBpe::BASE_SIZE
        texts = [texts] if texts.is_a?(String)
        Tempfile.create(["easy-ai-tokenizer", ".txt"]) do |file|
          texts.each { |text| file.puts(text) }
          file.flush
          trainer = ::Tokenizers::Trainers::BpeTrainer.new(vocab_size: vocab_size, min_frequency: min_frequency,
            special_tokens: SPECIALS, initial_alphabet: ::Tokenizers::PreTokenizers::ByteLevel.alphabet, show_progress: false)
          @backend.train([file.path], trainer)
        end
        self
      end

      def id(name)
        index = ByteBpe::SPECIALS.index(name.to_s)
        raise ArgumentError, "Unknown special token" unless index
        @backend.token_to_id(SPECIALS.fetch(index)) || raise(ArgumentError, "Tokenizer not trained")
      end

      def vocab_size
        @backend.vocab_size
      end

      def encode(text)
        validate_text!(text)
        @backend.encode(text, add_special_tokens: false).ids
      end

      def decode(ids, skip_special: true)
        @backend.decode(ids, skip_special_tokens: skip_special)
      end

      # The Ruby binding returns character offsets; normalize to byte intervals.
      def encode_with_offsets(text)
        validate_text!(text)
        encoded = @backend.encode(text, add_special_tokens: false)
        boundaries = [0]
        text.each_char { |character| boundaries << boundaries.last + character.bytesize }
        [encoded.ids, encoded.offsets.map { |start, stop| [boundaries.fetch(start), boundaries.fetch(stop)] }]
      end

      def to_h
        { "format" => "easy_ai.native_bpe", "version" => 1, "backend" => JSON.parse(@backend.to_s) }
      end

      def fingerprint
        Digest::SHA256.hexdigest(JSON.generate(canonical(to_h)))
      end

      def save(path)
        File.write(path, JSON.pretty_generate(to_h))
      end

      def self.from_h(data)
        raise ArgumentError, "Unsupported native tokenizer" unless data.values_at("format", "version") == ["easy_ai.native_bpe", 1]
        tokenizer = new(::Tokenizers::Tokenizer.from_str(JSON.generate(data.fetch("backend"))))
        ByteBpe::SPECIALS.each_with_index { |name, i| raise ArgumentError, "Special-token IDs mismatch" unless tokenizer.id(name) == i }
        tokenizer
      end

      private

      def validate_text!(text)
        raise ArgumentError, "Expected valid UTF-8 text" unless text.is_a?(String) && text.valid_encoding?
        # Added-token parsing cannot be disabled in this binding version.
        raise ArgumentError, "Text contains a reserved native tokenizer token" if SPECIALS.any? { |token| text.include?(token) }
      end

      def canonical(value)
        case value
        when Hash then value.keys.sort.to_h { |key| [key, canonical(value[key])] }
        when Array then value.map { |item| canonical(item) }
        else value
        end
      end
    end
  end
end
