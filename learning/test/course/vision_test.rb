require_relative "../test_helper"

class VisionTest < Minitest::Test
  def test_autoencoder_exact_linear_forward_and_sparse_loss
    model = EasyAILearning::Autoencoder::Model.new(input: 2, latent: 1, nonlinear: false)
    Torch.no_grad do
      model.encoder.weight.copy!(Torch.tensor([[1.0, 2.0]]))
      model.encoder.bias.zero!
      model.decoder.weight.copy!(Torch.tensor([[1.0], [3.0]]))
      model.decoder.bias.zero!
    end
    x = Torch.tensor([[1.0, 2.0]])

    assert_equal [[5.0, 15.0]], model.call(x).to_a
    expected = ((5 - 1)**2 + (15 - 2)**2) / 2.0 + 0.5 * 5

    assert_in_delta expected, model.objective(x, x, sparsity: 0.5).item, 1e-6
    assert_equal [[1.0, 0.0]], EasyAILearning::Autoencoder::Model.corrupt(x, mask: Torch.tensor([[true, false]])).to_a
  end

  def test_torch_convolution_matches_hand_calculation_and_derivative
    image, kernel = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]]
    conv = Torch::NN::Conv2d.new(1, 1, 2, bias: false)
    Torch.no_grad { conv.weight.copy!(Torch.tensor([[kernel]], dtype: :float32)) }
    x = Torch.tensor([[image]], dtype: :float32)
    expected = EasyAILearning::CNN::Convolution.correlate(image, kernel)

    assert_equal expected, conv.call(x).to_a[0][0]
    conv.call(x).sum.backward

    assert_equal [[12.0, 16.0], [24.0, 28.0]], conv.weight.grad.to_a[0][0]
    assert_equal 2, EasyAILearning::CNN::Convolution.output_size(4, 3, padding: 1, stride: 2)
  end

  def test_zero_residual_is_identity_and_gradient_has_direct_path
    block = EasyAILearning::Resnet::Block.new(input: 1, output: 1, normalize: false)
    Torch.no_grad { block.parameters.each(&:zero!) }
    x = Torch::NN::Parameter.new(Torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))
    output = block.call(x)

    assert_equal x.to_a, output.to_a
    output.sum.backward

    assert_equal [[[[1.0, 1.0], [1.0, 1.0]]]], x.grad.to_a
  end

  def test_projection_and_convolutional_autoencoder_shapes
    block = EasyAILearning::Resnet::Block.new(input: 2, output: 4, stride: 2)

    assert_equal [2, 4, 4, 4], block.call(Torch.ones([2, 2, 8, 8])).shape
    ae = EasyAILearning::Autoencoder::Convolutional.new

    assert_equal [2, 1, 8, 8], ae.call(Torch.ones([2, 1, 8, 8])).shape
    assert_equal [2, 4, 4, 4], ae.encode(Torch.ones([2, 1, 8, 8])).shape
  end

  def test_batchnorm_eval_does_not_change_running_state
    model = EasyAILearning::CNN::Model.new(normalize: true)
    x = Torch.ones([2, 1, 8, 8])
    model.train.call(x)
    model.eval
    before = model.state_dict.select { |k, _| k.include?("running") }.transform_values(&:clone)
    model.call(x * 10)

    before.each { |key, tensor| assert_equal tensor.to_a, model.state_dict[key].to_a }
  end
end

class ResidualVariantsTest < Minitest::Test
  def test_preactivation_zero_branch_preserves_negative_input_and_gradient
    block = EasyAILearning::Resnet::Preactivation.new(channels: 1).eval
    Torch.no_grad { block.parameters.each(&:zero!) }
    input = Torch::NN::Parameter.new(Torch.tensor([[[[-1.0, 2.0], [3.0, -4.0]]]]))
    output = block.call(input)
    output.sum.backward

    assert_equal input.to_a, output.to_a
    assert_equal [[[[1.0, 1.0], [1.0, 1.0]]]], input.grad.to_a
  end

  def test_bottleneck_projection_shape
    block = EasyAILearning::Resnet::Bottleneck.new(input: 4, output: 8, stride: 2)

    assert_equal [2, 8, 4, 4], block.call(Torch.ones([2, 4, 8, 8])).shape
  end
end
