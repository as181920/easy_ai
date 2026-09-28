require "test_helper"

class RotaryTest < Minitest::Test
  def test_rotations_preserve_norms_and_relative_attention_positions
    Torch.manual_seed(17)
    q = Torch.randn([2, 2, 5, 8], requires_grad: true)
    k = Torch.randn([2, 2, 5, 8], requires_grad: true)
    rotated = EasyAI::NN::RotaryPosition.call(q)
    keys = EasyAI::NN::RotaryPosition.call(k)
    shifted_q = EasyAI::NN::RotaryPosition.call(q, offset: 7)
    shifted_k = EasyAI::NN::RotaryPosition.call(k, offset: 7)

    assert_tensor_close((q * q).sum(dim: -1), (rotated * rotated).sum(dim: -1), 3e-6)
    assert_tensor_close(Torch.matmul(rotated, keys.transpose(-2, -1)), Torch.matmul(shifted_q, shifted_k.transpose(-2, -1)), 4e-6)
    rotated.sum.backward

    assert_predicate q.grad.abs.max.item, :finite?
    assert_operator q.grad.abs.sum.item, :>, 0
  end

  def test_rotary_model_preserves_padding_and_chunked_prediction
    config = tiny_config(model: { position_encoding: "rotary", pooling: "candidate", score_mode: "matching" })
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    model = EasyAI::Decision::ChoiceModel.new(config)
    row = example
    full = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu")
    chunked = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu", candidate_chunk_size: 1)
    expected = full.logits(row)

    expected.zip(chunked.logits(row)).each { |a, b| assert_in_delta a, b, 1e-6 }
    reversed = EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse))

    expected.zip(full.logits(reversed).reverse).each { |a, b| assert_in_delta a, b, 1e-6 }
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config).call([row, example(id: "long", state: "a longer state")])
    padded_logits = Torch.no_grad { model.call(batch)[0].to_a }

    expected.zip(padded_logits).each { |a, b| assert_in_delta a, b, 1e-6 }
    Torch::NN::Functional.cross_entropy(model.call(batch), batch[:targets]).backward

    assert model.parameters.all? { |parameter| parameter.grad.nil? || parameter.grad.abs.max.item.finite? }
    assert_raises(ArgumentError) { tiny_config(model: { hidden_size: 18, attention_heads: 2, position_encoding: "rotary" }) }
  end
end
