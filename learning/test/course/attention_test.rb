require_relative "../test_helper"

class AttentionTest < Minitest::Test
  def identity_attention
    model = EasyAILearning::Attention::MultiHead.new(embed_dim: 2)
    Torch.no_grad do
      [model.q_proj, model.k_proj, model.v_proj, model.o_proj].each do |layer|
        layer.weight.copy!(Torch.eye(2))
        layer.bias.zero!
      end
    end
    model
  end

  def test_attention_matches_manual_softmax_and_weighted_values
    model = identity_attention
    x = Torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    actual = model.call(x).to_a[0]
    probabilities = EasyAILearning::Foundations::Math.softmax([1 / Math.sqrt(2), 0])

    assert_in_delta probabilities[0], actual[0][0], 1e-6
    assert_in_delta probabilities[1], actual[0][1], 1e-6
    assert_in_delta 1, model.last_weights[0][0][0].sum.item, 1e-6
  end

  def test_causal_and_padding_masks_are_independent_and_block_leakage
    model = identity_attention
    x = Torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    before = model.call(x, causal: true)
    changed = x.clone
    changed[0][2] = Torch.tensor([50.0, -50.0])

    assert_equal before.narrow(1, 0, 2).to_a, model.call(changed, causal: true).narrow(1, 0, 2).to_a
    mask = Torch.tensor([[true, true, false]])
    first = model.call(x.narrow(1, 0, 1), memory: x, padding_mask: mask)
    second = model.call(x.narrow(1, 0, 1), memory: changed, padding_mask: mask)

    assert_equal first.to_a, second.to_a
    assert_raises(ArgumentError) { model.call(x, padding_mask: Torch.zeros([1, 3], dtype: :bool)) }
  end

  def test_encoder_without_positions_is_permutation_equivariant
    model = EasyAILearning::Transformer::SequenceModel.new(positions: false).eval
    a, b = Torch.tensor([[1, 2, 3]], dtype: :int64), Torch.tensor([[3, 2, 1]], dtype: :int64)
    first, second = model.call(a), model.call(b)

    assert_in_delta 0, (first[0][0] - second[0][2]).abs.max.item, 1e-6
    assert_in_delta 0, (first[0][1] - second[0][1]).abs.max.item, 1e-6
    assert_equal [0.0, 1.0, 0.0, 1.0], EasyAILearning::Transformer::Sinusoidal.values(1, 4).first
  end

  def test_encoder_decoder_and_gpt_cannot_read_future_decoder_tokens
    models = [EasyAILearning::Transformer::EncoderDecoder.new,
      EasyAILearning::GPT::Model.new(vocab_size: 7, block_size: 6, n_layer: 1, n_head: 2, n_embd: 8, dropout: 0)]
    models.each do |model|
      model.eval
      tokens = Torch.tensor([[1, 3, 4]], dtype: :int64)
      changed = Torch.tensor([[1, 3, 6]], dtype: :int64)
      a = model.is_a?(EasyAILearning::GPT::Model) ? model.call(tokens) : model.call(tokens, tokens)
      b = model.is_a?(EasyAILearning::GPT::Model) ? model.call(changed) : model.call(tokens, changed)

      assert_in_delta 0, (a.narrow(1, 0, 2) - b.narrow(1, 0, 2)).abs.max.item, 1e-6
    end
  end
end

class AdditiveAttentionTest < Minitest::Test
  def test_zero_alignment_scores_produce_mean_memory
    model = EasyAILearning::Attention::Additive.new(hidden: 2)
    Torch.no_grad { model.parameters.each(&:zero!) }
    memory = Torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = model.call(Torch.zeros([1, 2]), memory)

    assert_equal [[2.0, 3.0]], result.to_a
    assert_equal [[0.5, 0.5]], model.last_weights.to_a
  end

  def test_post_norm_and_pre_norm_zero_branches_have_different_semantics
    input = Torch.tensor([[[1.0, 3.0]]])
    pre = EasyAILearning::Transformer::EncoderBlock.new(width: 2, heads: 1)
    post = EasyAILearning::Transformer::EncoderBlock.new(width: 2, heads: 1, post_norm: true)
    [pre, post].each do |model|
      Torch.no_grad do
        model.named_parameters.each { |name, p| p.zero! unless name.include?("ln") }
      end
    end

    assert_equal input.to_a, pre.call(input).to_a
    assert_in_delta 0, post.call(input).mean.item, 1e-6
    assert_operator (post.call(input) - input).abs.max.item, :>, 1
  end
end
