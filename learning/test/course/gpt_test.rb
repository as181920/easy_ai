require_relative "../test_helper"

class GPTContractTest < Minitest::Test
  def setup
    @model = EasyAILearning::GPT::Model.new(vocab_size: 6, block_size: 3, n_layer: 1, n_head: 2, n_embd: 8, dropout: 0.2)
  end

  def test_generation_restores_training_mode_and_top_one_is_deterministic
    ids = Torch.tensor([[0, 1]], dtype: :int64)
    @model.train
    first = @model.generate(ids, max_new_tokens: 4, top_k: 1)
    second = @model.generate(ids, max_new_tokens: 4, top_k: 1)

    assert_equal [1, 6], first.shape
    assert_equal first.to_a, second.to_a
    assert @model.training
    assert_raises(ArgumentError) { @model.generate(ids, max_new_tokens: 1, temperature: 0) }
    assert_raises(ArgumentError) { @model.generate(ids, max_new_tokens: 1, top_k: 7) }
  end

  def test_batcher_right_shift_is_exact_and_seed_local
    tokenizer = EasyAILearning::Tokenizers::Character.new.train("abcde")
    dataset = EasyAILearning::GPT::TextDataset.new(tokenizer: tokenizer, text: "abcde", block_size: 3, auto_train: false)
    batcher = EasyAILearning::GPT::Batcher.new(dataset: dataset, batch_size: 2, seed: 7)
    x, y = batcher.next_batch

    assert_equal x.narrow(1, 1, 2).to_a, y.narrow(1, 0, 2).to_a
    assert_equal [2, 3], x.shape
  end
end
