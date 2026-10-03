require_relative "../test_helper"

class GenerativeTest < Minitest::Test
  def test_vae_reparameterization_and_kl_closed_form
    mean, logvar, noise = Torch.tensor([[1.0, 2.0]]), Torch.tensor([[0.0, Math.log(4.0)]]), Torch.tensor([[0.5, -0.5]])

    assert_equal [[1.5, 1.0]], EasyAILearning::Generative::Vae.reparameterize(mean, logvar, noise: noise).to_a
    expected = 0.5 * (1 + 4 + 1 + 4 - 2 - Math.log(4.0))

    assert_in_delta expected, EasyAILearning::Generative::Vae.kl(mean, logvar).item, 1e-6
    assert_in_delta 0, EasyAILearning::Generative::Vae.kl(Torch.zeros([1, 2]), Torch.zeros([1, 2])).item, 1e-7
  end

  def test_gan_discriminator_objective_does_not_backpropagate_into_generator
    model = EasyAILearning::Generative::Gan.new
    model.discriminator_loss(Torch.ones([2, 2]), model.call(Torch.ones([2, 2]))).backward

    assert model.generator.parameters.all? { |p| p.grad.nil? || p.grad.numel.zero? }
    assert model.discriminator.parameters.any? { |p| p.grad && p.grad.numel > 0 }
  end

  def test_diffusion_forward_and_final_reverse_match_equations
    model = EasyAILearning::Generative::Diffusion.new(count: 2, beta_start: 0.1, beta_end: 0.2)
    x, noise = Torch.tensor([[1.0, 2.0]]), Torch.tensor([[0.5, -0.5]])
    actual = model.q_sample(x, Torch.tensor([0], dtype: :int64), noise: noise).to_a.first

    assert_in_delta Math.sqrt(0.9) + Math.sqrt(0.1) * 0.5, actual.first, 1e-6
    a = model.reverse_step(x, time: 0, predicted_noise: noise, noise: Torch.ones_like(x))
    b = model.reverse_step(x, time: 0, predicted_noise: noise, noise: Torch.zeros_like(x))

    assert_equal a.to_a, b.to_a
    assert_in_delta (1 - 0.1 / Math.sqrt(0.1) * 0.5) / Math.sqrt(0.9), a.to_a.first.first, 1e-6
  end

  def test_patch_roundtrip_and_masked_input_invariance
    images = Torch.arange(64, dtype: :float32).reshape([1, 1, 8, 8])
    klass = EasyAILearning::Generative::MaskedAutoencoder

    assert_equal images.to_a, klass.unpatchify(klass.patchify(images)).to_a
    model = klass.new.eval
    original = Torch.zeros([1, 1, 8, 8])
    changed = original.clone
    changed[0][0][0][0] = 100
    visible = (1...16).to_a

    assert_equal model.call(original, visible: visible).to_a, model.call(changed, visible: visible).to_a
    assert_raises(ArgumentError) { model.objective(original, visible: (0...16).to_a) }
  end
end
