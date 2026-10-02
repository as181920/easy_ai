require_relative "../test_helper"

class LearningGPTModelTest < Minitest::Test
  def test_reorganized_gpt_can_train_and_generate_with_the_shared_components
    Torch.manual_seed(1337)
    model = EasyAILearning::GPT::Model.new(vocab_size: 12, block_size: 4, n_layer: 1, n_head: 2, n_embd: 8, dropout: 0.0)
    input = Torch.tensor([[1, 2, 3]], dtype: :int64)
    before = model.parameters.last.clone
    optimizer = Torch::Optim::AdamW.new(model.parameters, lr: 0.01)
    logits = model.call(input)
    logits.square.mean.backward
    optimizer.step

    assert_equal [1, 3, 12], logits.shape
    assert_operator (before - model.parameters.last).abs.max.item, :>, 0
    assert_equal [1, 5], model.generate(input, max_new_tokens: 2).shape
  end
end
