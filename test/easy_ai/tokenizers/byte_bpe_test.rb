require "test_helper"

describe EasyAI::Tokenizers::ByteBpe do
  before do
    @en_corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
    data_path = File.expand_path("../../../data/xiaojing.txt", __dir__)
    @cjk_corpus = File.read(data_path)
  end

  describe "train" do
    it "trains tokenizer" do
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      tokenizer.train(@cjk_corpus)

      assert_includes tokenizer.merges.values, ("之".dup).force_encoding(Encoding::ASCII_8BIT)
      assert_includes tokenizer.merges.values, ("。".dup).force_encoding(Encoding::ASCII_8BIT)
    end
  end

  describe "tokenize" do
    before do
      @tokenizer = EasyAI::Tokenizers::ByteBpe.new
      @tokenizer.train(@cjk_corpus)
    end

    it "handles CJK words" do
      result = @tokenizer.tokenize("你好")

      assert_equal "你好<|w|>", result.join.force_encoding("utf-8")
    end
  end

  describe "detokenize" do
    before do
      @tokenizer = EasyAI::Tokenizers::ByteBpe.new
    end

    it "detokenizes tokens" do
      assert_equal "你好", @tokenizer.detokenize(["\xE4", "\xBD", "\xA0", "\xE5", "\xA5", "\xBD", "<|w|>"])
    end
  end
end
