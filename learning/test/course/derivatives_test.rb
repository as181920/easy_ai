require_relative "../test_helper"

module CourseDerivativeChecks
  # Check selected entries independently with central differences, without optimizing.
  def check_gradients(model, objective)
    model.eval
    objective.call.backward
    model.named_parameters.each do |name, parameter|
      next unless parameter.requires_grad
      analytic = parameter.grad.reshape([-1])[0].item
      flat = parameter.reshape([-1])
      original = flat[0].item
      epsilon = 0.002
      Torch.no_grad { flat[0] = original + epsilon }
      plus = objective.call.item
      Torch.no_grad { flat[0] = original - epsilon }
      minus = objective.call.item
      Torch.no_grad { flat[0] = original }

      assert_in_delta analytic, (plus - minus) / (2 * epsilon), 8e-4, "#{model.class} #{name}"
    end
  end
end

class DerivativesTest < Minitest::Test
  include CourseDerivativeChecks

  def test_lstm_and_gru_gate_derivatives
    %i[lstm gru].each do |kind|
      Torch.manual_seed(7)
      cell = EasyAILearning::RNN::Cell.new(input: 2, hidden: 2, kind: kind)
      x = Torch.tensor([[0.2, -0.4]])
      h = Torch.tensor([[0.1, 0.3]])
      state = kind == :lstm ? [h, Torch.tensor([[0.3, -0.1]])] : h
      check_gradients(cell, -> { result = cell.call(x, state); (result.is_a?(Array) ? result.first : result).square.sum })
    end
  end

  def test_attention_and_transformer_derivatives
    Torch.manual_seed(12)
    attention = EasyAILearning::Attention::MultiHead.new(embed_dim: 2)
    x = Torch.tensor([[[0.3, -0.2], [0.1, 0.4]]])
    check_gradients(attention, -> { attention.call(x).square.mean })
    Torch.manual_seed(12)
    block = EasyAILearning::Transformer::EncoderBlock.new(width: 2, heads: 1)
    check_gradients(block, -> { block.call(x).square.mean })
  end

  def test_variational_reconstruction_and_kl_derivatives
    Torch.manual_seed(12)
    model = EasyAILearning::Generative::Vae.new(input: 2, latent: 1)
    x, noise = Torch.tensor([[0.3, -0.2]]), Torch.tensor([[0.4]])
    check_gradients(model, -> { model.objective(x, noise: noise) })
  end
end

class FurtherDerivativesTest < Minitest::Test
  include CourseDerivativeChecks
  def test_seq2seq_and_decoder_gradient_chain
    Torch.manual_seed(19)
    model = EasyAILearning::Seq2seq::Model.new(vocab: 7, hidden: 2, attention: true)
    source = Torch.tensor([[3, 4]], dtype: :int64)
    decoder = Torch.tensor([[1, 4]], dtype: :int64)
    check_gradients(model, -> { model.call(source, decoder).square.mean })
  end

  def test_gan_generator_and_discriminator_gradient_chain
    Torch.manual_seed(19)
    model = EasyAILearning::Generative::Gan.new
    noise = Torch.tensor([[0.2, -0.4]])
    check_gradients(model, -> { model.generator_loss(noise) })
  end

  def test_diffusion_noise_predictor_derivatives
    Torch.manual_seed(19)
    model = EasyAILearning::Generative::Diffusion.new(count: 2)
    x, time, noise = Torch.tensor([[0.2, -0.4]]), Torch.tensor([1], dtype: :int64), Torch.tensor([[0.1, -0.2]])
    check_gradients(model, -> { model.objective(x, time: time, noise: noise) })
  end
end
