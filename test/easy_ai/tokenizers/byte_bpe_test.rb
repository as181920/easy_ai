require "test_helper"

class ByteBpeTest < Minitest::Test
  def test_unseen_scripts_and_whitespace_round_trip
    tokenizer = EasyAI::Tokenizers::ByteBpe.new.train(["hello hello 中文 中文"], vocab_size: 300)
    text = "你好\t世界\n  مرحبا Привет 🌈 e\u0301 [PAD]"

    assert_equal text, tokenizer.decode(tokenizer.encode(text))
    refute_includes tokenizer.encode(text), tokenizer.id(:unk)
  end

  def test_save_and_load_preserve_ids_and_merges
    tokenizer = EasyAI::Tokenizers::ByteBpe.new.train(["abcabc hello hello"], vocab_size: 300)
    Dir.mktmpdir do |dir|
      path = File.join(dir, "tokenizer.json")
      tokenizer.save(path)
      restored = EasyAI::Tokenizers::Registry.load(path)

      assert_equal tokenizer.encode("abcabc 未知字"), restored.encode("abcabc 未知字")
      assert_equal tokenizer.fingerprint, restored.fingerprint
      assert_equal tokenizer.merges, restored.merges
    end
  end

  def test_native_backend_is_self_trained_and_round_trips
    tokenizer = EasyAI::Tokenizers::NativeBpe.new.train(["hello 世界 مرحبا " * 4], vocab_size: 300)
    text = "new\n中文字 🌈 مرحبا"

    assert_equal text, tokenizer.decode(tokenizer.encode(text))
    restored = EasyAI::Tokenizers::NativeBpe.from_h(tokenizer.to_h)

    assert_equal tokenizer.fingerprint, restored.fingerprint
    assert_equal 0, tokenizer.id(:pad)
  end

  def test_invalid_merges_are_rejected
    data = EasyAI::Tokenizers::ByteBpe.new.to_h.merge("merges" => [[999, 6]])
    assert_raises(ArgumentError) { EasyAI::Tokenizers::ByteBpe.from_h(data) }
  end
end
