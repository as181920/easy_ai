require_relative "../test_helper"
require_relative "../byte_bpe_tokenizer"

describe ByteBpeTokenizer do
  before do
    @corpus = File.read File.expand_path("../data/xiaojing.txt", __dir__)
  end

  describe "train" do
    it "should train tokenizer" do
      tokenizer = ByteBpeTokenizer.new
      tokenizer.train(@corpus)

      assert_includes tokenizer.merges.values, (+"之").force_encoding(Encoding::ASCII_8BIT)
      assert_includes tokenizer.merges.values, (+"。").force_encoding(Encoding::ASCII_8BIT)
    end
  end

  describe "tokenize" do
    before do
      @tokenizer = ByteBpeTokenizer.new
      @tokenizer.train(@corpus)
    end

    it "should handle CJK words" do
      result = @tokenizer.tokenize("你好")

      assert_equal "你好<|w|>", result.join.force_encoding("utf-8")
    end
  end

  describe "detokenize" do
    before do
      @tokenizer = ByteBpeTokenizer.new
    end

    it "should detokenize tokens" do
      assert_equal "你好", @tokenizer.detokenize(["\xE4", "\xBD", "\xA0", "\xE5", "\xA5", "\xBD", "<|w|>"])
    end
  end
end
