require "test_helper"

describe EasyAI::Tokenizers::WordBpe do
  describe "train" do
    it "trains tokenizer" do
      tokenizer = EasyAI::Tokenizers::WordBpe.new
      corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
      tokenizer.train(corpus)

      assert_includes tokenizer.merges, ["new", "est<|w|>"]
      assert_includes tokenizer.merges, ["w", "i"]
    end
  end

  describe "tokenize" do
    before do
      @tokenizer = EasyAI::Tokenizers::WordBpe.new
      corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
      @tokenizer.train(corpus)
    end

    it "tokenizes english words" do
      result = @tokenizer.tokenize("lower")

      assert_equal ["low", "e", "r", "<|w|>"], result
    end

    it "handles CJK words" do
      result = @tokenizer.tokenize("你好")

      assert_equal ["你", "<|w|>", "好", "<|w|>"], result
    end
  end

  describe "detokenize" do
    before do
      @tokenizer = EasyAI::Tokenizers::WordBpe.new
    end

    it "detokenizes tokens" do
      tokens = ["low", "e", "r", "<|w|>"]

      assert_equal "lower", @tokenizer.detokenize(tokens)
      assert_equal "你好", @tokenizer.detokenize(["你", "<|w|>", "好", "<|w|>"])
    end
  end
end
