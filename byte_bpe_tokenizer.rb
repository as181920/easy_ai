require "logger"
require "active_support/all"

class ByteBpeTokenizer
  PAIR_SIZE = 2
  END_OF_WORD_TOKEN = "<|w|>".b.freeze
  BYTE_SYMBOLS = (0..255).map { |byte| byte.chr(Encoding::ASCII_8BIT) }.freeze

  attr_accessor :num_merges, :min_freq, :logger

  attr_reader :merges, :vocab

  def initialize(num_merges: 10, min_freq: 2)
    @num_merges = num_merges
    @min_freq = min_freq
    @merges = {}
    @vocab = Hash.new(0)
    @logger = ActiveSupport::Logger.new(
      ENV.fetch("LOG_PATH", STDOUT),
      level: ENV.fetch("LOG_LEVEL", "DEBUG")
    )
  end

  def train(text)
    corpus = build_corpus(text)

    num_merges.times do |nth_merge|
      pair, frequency = most_frequent_pair(corpus)
      break unless pair
      break if frequency < min_freq

      merges[pair.dup] = replacement = pair.join
      corpus = merge_corpus(corpus, pair, replacement)
      logger.debug("Iteration #{nth_merge.succ}: merged #{pair.inspect} -> #{replacement.inspect} (freq=#{frequency})")
    end

    rebuild_vocab!(corpus)
  end

  def tokenize(text)
    build_corpus(text).flat_map do |tokens|
      apply_all_merges(tokens.dup)
    end
  end

  def detokenize(tokens)
    words = []
    buffer = []

    tokens.each do |token|
      if token == END_OF_WORD_TOKEN
        next if buffer.empty?

        words << decode_word(buffer)
        buffer.clear
      else
        buffer << token
      end
    end

    words << decode_word(buffer) unless buffer.empty?
    words.join(" ")
  end

  private

    def build_corpus(text)
      split_words(text).map { |word| encode_word(word) + [END_OF_WORD_TOKEN] }
    end

    def split_words(text)
      text.scan(/\S+/)
    end

    def encode_word(word)
      word.encode("UTF-8").bytes.map { |byte| BYTE_SYMBOLS[byte] }
    end

    def most_frequent_pair(corpus)
      counts = Hash.new(0)

      corpus.each do |tokens|
        next if tokens.length < PAIR_SIZE

        tokens.each_cons(PAIR_SIZE) { |pair| counts[pair.dup] += 1 }
      end

      counts.max_by { |_, freq| freq }
    end

    def merge_corpus(corpus, pair, replacement)
      corpus.map { |tokens| apply_merge(tokens.dup, pair, replacement) }
    end

    def apply_merge(tokens, pair, replacement)
      index = 0
      while index < tokens.length - 1
        if tokens[index] == pair[0] && tokens[index + 1] == pair[1]
          tokens[index, PAIR_SIZE] = [replacement]
          index = [index - 1, 0].max
        else
          index += 1
        end
      end
      tokens
    end

    def apply_all_merges(tokens)
      merges.each do |pair, replacement|
        tokens = apply_merge(tokens, pair, replacement)
      end
      tokens
    end

    def rebuild_vocab!(corpus)
      @vocab = Hash.new(0)
      corpus.each { |tokens| @vocab[tokens.join(" ")] += 1 }
    end

    def decode_word(byte_tokens)
      byte_tokens.join.force_encoding(Encoding::UTF_8)
    rescue Encoding::UndefinedConversionError, Encoding::InvalidByteSequenceError
      byte_tokens.join.force_encoding(Encoding::ASCII_8BIT)
    end
end
