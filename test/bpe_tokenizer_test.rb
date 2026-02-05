require_relative "../test_helper"
require_relative "../bpe_tokenizer"

describe BpeTokenizer do
  # before do
  # end

  describe "train" do
    it "should train tokenizer" do
      tokenizer = BpeTokenizer.new
      corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
      tokenizer.train(corpus)

      assert_includes tokenizer.merges, ["new", "est<|w|>"]
      assert_includes tokenizer.merges, ["w", "i"]
    end
  end

  describe "tokenize" do
    before do
      @tokenizer = BpeTokenizer.new
      corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
      @tokenizer.train(corpus)
    end

    it "should tokenize english word" do
      result = @tokenizer.tokenize("lower")

      assert_equal ["low", "e", "r", "<|w|>"], result
    end

    it "should handle CJK words" do
      result = @tokenizer.tokenize("你好")

      assert_equal ["你", "好", "<|w|>"], result
    end
  end
end
