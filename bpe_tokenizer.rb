require "debug"
require "active_support/all"

class BpeTokenizer
  PAIR_SIZE = 2
  END_OF_WORD_TOKEN = "<|w|>".freeze

  attr_accessor :num_merges, :min_freq, :logger

  attr_reader :merges, :vocab

  # num_merges: 2k~5k for small text; 10k~30k for medium; 30k~150k for large text
  # min_freq: 2~5 for small text; 5~10 for medium; 10+ for large text
  def initialize(num_merges: 10, min_freq: 2)
    @num_merges = num_merges
    @min_freq = min_freq
    @merges = {}
    @vocab = []
    @logger = ActiveSupport::Logger.new(STDOUT, level: ENV.fetch("LOG_LEVEL", "DEBUG"))
  end

  def train(text)
    words = text.split.each_with_object(Hash.new(0)) do |word, counts|
      counts[word.chars + [END_OF_WORD_TOKEN]] += 1
    end

    num_merges.times do |nth_merge|
      pair_stats = get_stats(words)
      return if pair_stats.empty?

      best_pair = pair_stats.max_by { |_, v| v }.first
      @merges[best_pair] = best_pair.join

      words = merge_vocab(words, best_pair)
      logger.info "Iteration #{nth_merge.succ}: Merging #{best_pair} -> #{merges[best_pair]}"
    end
  end

  def tokenize(text)
    text.split.map do |word|
      word_tokens = word.chars + [END_OF_WORD_TOKEN]
      merges.each do |pair, _replacement|
        word_tokens = apply_merge(word_tokens, pair)
      end
      word_tokens
    end
  end

  private

    def get_stats(words)
      stats = Hash.new(0)
      words.each do |word_tokens, count|
        word_tokens.each_cons(PAIR_SIZE).with_object(stats) do |pair, stats|
          stats[pair] += count
        end
      end
      stats
    end

    def merge_vocab(words, pair)
      words.each do |word_tokens, _count|
        word_tokens = apply_merge(word_tokens, pair)
      end
    end

    def apply_merge(tokens, pair)
      index = tokens.each_cons(PAIR_SIZE).with_index.find { |sub_tokens, _| sub_tokens == pair }&.last
      return tokens unless index

      tokens[index, PAIR_SIZE] = [merges[pair]]
      tokens
    end
end
