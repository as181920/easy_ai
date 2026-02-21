module EasyAI
  module Tokenizers
    class WordBpe < BaseTokenizer
      PAIR_SIZE = 2
      END_OF_WORD_TOKEN = "<|w|>".freeze
      CJK_CHAR_REGEX = /[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]/.freeze
      SPACE_BEFORE_CJK = Regexp.new("\\s+(?=#{CJK_CHAR_REGEX.source})").freeze
      SPACE_AFTER_CJK = Regexp.new("(?<=#{CJK_CHAR_REGEX.source})\\s+").freeze

      attr_accessor :num_merges, :min_freq
      attr_reader :merges, :vocab

      def initialize(num_merges: 10, min_freq: 2, **kwargs)
        super(**kwargs)
        @num_merges = num_merges
        @min_freq = min_freq
        @merges = {}
        @vocab = Hash.new(0)
      end

      def train(text)
        merges.clear
        words = pre_tokenize(text).split.each_with_object(Hash.new(0)) do |word, counts|
          counts[word.chars + [END_OF_WORD_TOKEN]] += 1
        end

        num_merges.times do |nth_merge|
          pair_stats = get_stats(words)
          break if pair_stats.empty?

          best_pair, frequency = pair_stats.max_by { |_, v| v }
          break if frequency < min_freq

          best_pair = best_pair.dup
          merges[best_pair] = replacement = best_pair.join
          words = merge_vocab(words, best_pair, replacement)
          logger.debug "Iteration #{nth_merge.succ}: merging #{best_pair.inspect} -> #{replacement.inspect}"
        end

        rebuild_vocab!(words)
      end

      def tokenize(text)
        logger.info "#{self.class} tokenize text: #{text}"

        word_tokens = pre_tokenize(text).split.map do |word|
          tokens = word.chars + [END_OF_WORD_TOKEN]
          merges.each { |pair, replacement| tokens = apply_merge(tokens, pair, replacement) }
          tokens
        end

        result = word_tokens.flatten
        logger.info "#{self.class} tokenize result: #{result}"
        result
      end

      def detokenize(tokens)
        logger.info "#{self.class} detokenize tokens: #{tokens}"

        words = []
        buffer = []

        tokens.each do |token|
          if token == END_OF_WORD_TOKEN
            next if buffer.empty?

            words << buffer.join
            buffer.clear
          else
            buffer << token
          end
        end
        words << buffer.join unless buffer.empty?

        result = post_detokenize(words.join(" ")).strip
        logger.info "#{self.class} detokenize result: #{result}"
        result
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

        def merge_vocab(words, pair, replacement)
          new_words = Hash.new(0)
          words.each do |word_tokens, count|
            merged = apply_merge(word_tokens.dup, pair, replacement)
            new_words[merged] += count
          end
          new_words
        end

        def apply_merge(tokens, pair, replacement)
          loop do
            index = tokens.each_cons(PAIR_SIZE).with_index.find { |sub_tokens, _| sub_tokens == pair }&.last
            break unless index

            tokens[index, PAIR_SIZE] = [replacement]
          end

          tokens
        end

        def pre_tokenize(text)
          text.gsub(CJK_CHAR_REGEX) { |char| " #{char} " }
        end

        def post_detokenize(text)
          text.gsub(SPACE_BEFORE_CJK, "").gsub(SPACE_AFTER_CJK, "")
        end

        def rebuild_vocab!(words)
          @vocab = Hash.new(0)

          words.each do |word_tokens, count|
            word_tokens.each do |token|
              next if token == END_OF_WORD_TOKEN

              @vocab[token] += count
            end
          end

          all_tokens = (@vocab.keys + [END_OF_WORD_TOKEN]).uniq
          rebuild_token_mappings!([all_tokens])
        end
    end
  end
end
