require "debug"

class BPETokenizer
  attr_reader :merges, :vocab

  def initialize
    @merges = {}
    @vocab = []
  end

  # 1. 训练模型：从语料库学习合并规则
  def train(text, num_merges)
    # 初始化：将文本拆分为字符数组（模拟字节级或字符级基础词表）
    # 使用 Hash 记录词频以提高统计效率
    words = text.split.each_with_object(Hash.new(0)) do |word, counts|
      counts[word.chars + ["</w>"]] += 1 # </w> 为词尾标记
    end

    num_merges.times do |i|
      pairs = get_stats(words)
      break if pairs.empty?

      # 寻找出现频率最高的一对
      best_pair = pairs.max_by { |_, v| v }.first
      @merges[best_pair] = best_pair.join("")

      # 执行合并
      words = merge_vocab(best_pair, words)
      puts "Iteration #{i + 1}: Merging #{best_pair} -> #{@merges[best_pair]}"
    end
  end

  # 2. 编码：将新文本根据学到的规则进行分词
  def tokenize(text)
    text.split.map do |word|
      tokens = word.chars + ["</w>"]
      @merges.each do |pair, replacement|
        tokens = apply_merge(tokens, pair, replacement)
      end
      tokens
    end.flatten
  end

  private

  # 统计相邻符号对的频率
  def get_stats(words)
    pairs = Hash.new(0)
    words.each do |word_tokens, count|
      (word_tokens.length - 1).times do |i|
        pair = [word_tokens[i], word_tokens[i+1]]
        pairs[pair] += count
      end
    end
    pairs
  end

  # 在训练语料中执行合并
  def merge_vocab(pair, words)
    new_words = {}
    replacement = pair.join("")
    words.each do |word_tokens, count|
      new_tokens = apply_merge(word_tokens, pair, replacement)
      new_words[new_tokens] = count
    end
    new_words
  end

  # 在单个标记序列中替换指定的符号对
  def apply_merge(tokens, pair, replacement)
    new_tokens = []
    i = 0
    while i < tokens.length
      if i < tokens.length - 1 && tokens[i] == pair[0] && tokens[i+1] == pair[1]
        new_tokens << replacement
        i += 2
      else
        new_tokens << tokens[i]
        i += 1
      end
    end
    new_tokens
  end
end

# --- 测试代码 ---
corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
tokenizer = BPETokenizer.new
tokenizer.train(corpus, 10)

puts "\nTokenizing 'lower':"
p tokenizer.tokenize("lower")
