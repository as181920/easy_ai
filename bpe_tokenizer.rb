require "debug"
require "active_support/all"

class BpeTokenizer
  PAIR_SIZE = 2
  END_OF_WORD_TOKEN = "<|w|>".freeze
  CJK_CHAR_REGEX = /[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]/.freeze

  attr_accessor :num_merges, :min_freq, :logger

  attr_reader :merges, :vocab

  # num_merges: 2k~5k for small text; 10k~30k for medium; 30k~150k for large text
  # min_freq: 2~5 for small text; 5~10 for medium; 10+ for large text
  def initialize(num_merges: 10, min_freq: 2)
    @num_merges = num_merges
    @min_freq = min_freq
    @merges = {}
    @vocab = []
    @logger = ActiveSupport::Logger.new(
      ENV.fetch("LOG_PATH", STDOUT),
      level: ENV.fetch("LOG_LEVEL", "DEBUG")
    )
  end

  def train(text)
    words = pre_tokenize(text).split.each_with_object(Hash.new(0)) do |word, counts|
      counts[word.chars + [END_OF_WORD_TOKEN]] += 1
    end

    num_merges.times do |nth_merge|
      pair_stats = get_stats(words)
      return if pair_stats.empty?

      best_pair, frequency = pair_stats.max_by { |_, v| v }
      break if frequency < min_freq

      @merges[best_pair] = replacement = best_pair.join

      words = merge_vocab(words, best_pair, replacement)
      logger.debug "Iteration #{nth_merge.succ}: Merging #{best_pair} -> #{merges[best_pair]}"
    end
  end

  def tokenize(text)
    logger.info "#{self.class} tokenize text: #{text}"

    word_tokens = pre_tokenize(text).split.map do |word|
      (word.chars + [END_OF_WORD_TOKEN]).tap do |word_tokens|
        merges.each { |pair, replacement| word_tokens = apply_merge(word_tokens, pair, replacement) }
      end
    end

    word_tokens.flatten.tap { logger.info "#{self.class} tokenize result: #{_1}" }
  end

  private

    def get_stats(words)
      stats = Hash.new(0)
      words.each do |word_tokens, count|
        next if word_tokens.length < PAIR_SIZE

        word_tokens.each_cons(PAIR_SIZE) { |pair| stats[pair] += count }
      end
      stats
    end

    def merge_vocab(words, pair, replacement = nil)
      words.each { |word_tokens, _count| word_tokens = apply_merge(word_tokens, pair, replacement || merges[pair]) }
    end

    def apply_merge(tokens, pair, replacement = nil)
      index = tokens.each_cons(PAIR_SIZE).with_index.find { |sub_tokens, _| sub_tokens == pair }&.last
      return tokens unless index

      tokens[index, PAIR_SIZE] = [replacement || merges[pair]]
      tokens
    end

    def pre_tokenize(text)
      text.gsub(CJK_CHAR_REGEX) { |char| " #{char} " }
    end
end
