require "set"
require "unicode_normalize"

# A lightweight, pure-Ruby approximation of SentencePiece's Unigram tokenizer.
# It normalizes text, prepends the "▁" (U+2581) boundary marker, builds a
# candidate vocabulary, then runs a few pruning iterations using Viterbi
# segmentation to keep the most useful pieces.
class SentencePieceTokenizer
  attr_reader :vocab, :scores

  DEFAULT_MAX_PIECE_LENGTH = 6

  def initialize(vocab_size: 8000, max_piece_length: DEFAULT_MAX_PIECE_LENGTH, pruning_factor: 2, em_iterations: 5)
    @vocab_size = vocab_size
    @max_piece_length = max_piece_length
    @pruning_factor = pruning_factor
    @em_iterations = em_iterations
    @vocab = []
    @scores = {}
    @base_pieces = []
  end

  def train(text)
    sentences = normalize_corpus(text)
    raise ArgumentError, "no sentences to train on" if sentences.empty?

    initialize_vocab(sentences)
    em_iterations.times { expectation_maximization!(sentences) }
    @vocab = @pieces
    self
  end

  def tokenize(text)
    ensure_trained!
    normalize_corpus(text).flat_map { |sentence| segment(sentence) }
  end

  private

    attr_reader :vocab_size, :max_piece_length, :pruning_factor, :em_iterations

    def ensure_trained!
      raise "Call #train before tokenizing" if @pieces.nil? || @pieces.empty?
    end

    def normalize_corpus(text)
      text.each_line.map { |line| normalize_sentence(line) }.reject(&:empty?)
    end

    def normalize_sentence(sentence)
      normalized = sentence.unicode_normalize(:nfkc)
      normalized.gsub!(/\s+/, " ")
      normalized.strip!
      return "" if normalized.empty?

      "▁" + normalized.gsub(" ", "▁")
    end

    def initialize_vocab(sentences)
      counts = Hash.new(0)
      single_chars = Set.new

      sentences.each do |sentence|
        chars = sentence.chars
        chars.each do |char|
          counts[char] += 1
          single_chars << char
        end

        0.upto(chars.length - 1) do |start|
          2.upto([max_piece_length, chars.length - start].min) do |length|
            piece = chars.slice(start, length).join
            counts[piece] += 1
          end
        end
      end

      sorted_candidates = counts.sort_by { |_, freq| -freq }.map(&:first)
      @base_pieces = single_chars.to_a
      @pieces = (@base_pieces + sorted_candidates).uniq.first(vocab_size * pruning_factor)
      initialize_scores!
    end

    def initialize_scores!
      uniform_score = -Math.log(1.0 / @pieces.size)
      @scores = @pieces.map { |piece| [piece, uniform_score] }.to_h
    end

    def expectation_maximization!(sentences)
      frequencies = Hash.new(0)

      sentences.each do |sentence|
        segment(sentence).each { |piece| frequencies[piece] += 1 }
      end

      prune_vocab!(frequencies)
      update_scores!(frequencies)
    end

    def segment(sentence)
      chars = sentence.chars
      n = chars.length
      best_score = Array.new(n + 1, Float::INFINITY)
      best_score[0] = 0.0
      best_prev = Array.new(n + 1)

      0.upto(n - 1) do |idx|
        next if best_score[idx].infinite?

        max_len = [max_piece_length, n - idx].min
        max_len.downto(1) do |length|
          piece = chars.slice(idx, length).join
          score = scores[piece]
          next unless score

          candidate_score = best_score[idx] + score
          next unless candidate_score < best_score[idx + length]

          best_score[idx + length] = candidate_score
          best_prev[idx + length] = [idx, piece]
        end
      end

      return chars if best_prev[n].nil?

      reconstruct_tokens(best_prev, chars)
    end

    def reconstruct_tokens(best_prev, chars)
      tokens = []
      idx = chars.length

      while idx > 0
        prev = best_prev[idx]
        if prev.nil?
          tokens.unshift(chars[idx - 1])
          idx -= 1
          next
        end

        start_idx, piece = prev
        tokens.unshift(piece)
        idx = start_idx
      end

      tokens
    end

    def prune_vocab!(frequencies)
      keep = @base_pieces.dup

      ranked = frequencies.sort_by { |piece, freq| [-freq, piece.length] }
      ranked.each do |piece, _|
        next if keep.include?(piece)
        keep << piece
        break if keep.size >= vocab_size
      end

      @pieces = keep.first(vocab_size)
      @pieces.each { |piece| frequencies[piece] ||= 1 }
    end

    def update_scores!(frequencies)
      total = @pieces.sum { |piece| frequencies.fetch(piece, 1) }
      @scores = @pieces.each_with_object({}) do |piece, acc|
        freq = frequencies.fetch(piece, 1)
        prob = freq.to_f / total
        acc[piece] = -Math.log(prob)
      end
    end
end
