module EasyAILearning
  module Generative
    module Experiment
      module_function

      def run(c)
        manifold, valid = Course::Data.manifold(seed: c.seed), Course::Data.manifold(seed: c.seed + 1)
        mixture = Course::Data.mixture(seed: c.seed)
        c.artifacts.json("data", { train_manifold: manifold, validation_manifold: valid, train_mixture: mixture,
          images: Course::Data.images(seed: c.seed).first, tokens: Course::Data.sequences(seed: c.seed).first })
        x, vx = c.tensor(manifold), c.tensor(valid)
        vae = c.model(Vae.new)
        c.train("vae", vae, validation: -> { Torch::NN::Functional.mse_loss(vae.call(vx), vx) }) { vae.objective(x, beta: 0.1) }
        c.results[:vae] = c.evaluate(vae) { mu, logvar = vae.distribution(vx); { reconstruction_mse: Torch::NN::Functional.mse_loss(vae.call(vx), vx).item, kl: Vae.kl(mu, logvar).item } }
        c.artifacts.json("vae-prior-samples", c.evaluate(vae) { vae.decoder.call(Torch.randn([64, 2], device: c.device)).cpu.to_a })
        c.save("vae-model", vae)
        train_gan(c, c.tensor(mixture))
        diffusion = c.model(Diffusion.new)
        real = c.tensor(mixture)
        c.train("diffusion", diffusion) do
          diffusion.objective(real, time: Torch.randint(12, [real.shape[0]], dtype: :int64, device: c.device), noise: Torch.randn_like(real))
        end
        samples = diffusion.sample(device: c.device)
        c.results[:diffusion] = { sample_stats: Diagnostics::Stats.activation(samples) }
        c.artifacts.json("diffusion-samples", samples.cpu.to_a)
        c.artifacts.scatter("diffusion-distribution", { target: mixture, generated: samples.cpu.to_a })
        c.save("diffusion-model", diffusion)
        train_masked(c)
      end

      def train_gan(c, real)
        gan = c.model(Gan.new)
        d = Training::Optimizer.new(gan.discriminator.named_parameters)
        g = Training::Optimizer.new(gan.generator.named_parameters)
        history = []
        c.steps.times do |step|
          d.zero_grad
          noise = Torch.randn([real.shape[0], 2], device: c.device)
          dl = gan.discriminator_loss(real, gan.call(noise))
          dl.backward
          d.step
          g.zero_grad
          gl = gan.generator_loss(noise)
          gl.backward
          g.step
          history << [step + 1, dl.item, gl.item]
        end
        c.artifacts.json("gan-history", history)
        c.artifacts.plot("gan-loss", { discriminator: history.map { |s, dloss, _| [s, dloss] }, generator: history.map { |s, _, gloss| [s, gloss] } })
        samples = c.evaluate(gan) { gan.call(Torch.randn([64, 2], device: c.device)).cpu.to_a }
        c.artifacts.json("gan-samples", samples)
        c.artifacts.scatter("gan-distribution", { target: real.cpu.to_a, generated: samples })
        c.results[:gan] = { discriminator_loss: history.last[1], generator_loss: history.last[2], sample_stats: Diagnostics::Stats.summarize(samples) }
        c.save("gan-model", gan)
      end

      def train_masked(c)
        rows, labels = Course::Data.sequences(seed: c.seed)
        x = c.tensor(rows, integer: true)
        y = x.clone
        model = c.model(Transformer::SequenceModel.new(vocab: 8))
        # Same fixed mask for validation; random masks for training. Token 7 is MASK.
        fixed = Torch.zeros(x.shape, dtype: :bool, device: c.device)
        fixed[0..-1, 2] = true
        validation_rows, = Course::Data.sequences(seed: c.seed + 1)
        vx = c.tensor(validation_rows, integer: true)
        c.train("masked-language", model, validation: -> { Training::Math.masked_cross_entropy(model.call(vx.masked_fill(fixed, 7)), vx, mask: fixed) }) do
          mask = Torch.rand(x.shape, device: c.device).lt(0.3)
          mask[0..-1, 2] = true
          Training::Math.masked_cross_entropy(model.call(x.masked_fill(mask, 7)), y, mask: mask)
        end
        c.results[:masked_language] = c.evaluate(model) { { loss: Training::Math.masked_cross_entropy(model.call(vx.masked_fill(fixed, 7)), vx, mask: fixed).item } }
        c.save("masked-language-model", model, config: { vocab: 8 })
        images, = Course::Data.images(seed: c.seed)
        valid_images, = Course::Data.images(seed: c.seed + 1)
        ix, iv = c.tensor(images), c.tensor(valid_images)
        mae = c.model(MaskedAutoencoder.new)
        rng = Random.new(c.seed)
        visible = (0...16).to_a.sample(8, random: rng).sort
        c.train("mae", mae, validation: -> { mae.objective(iv, visible: visible) }) do
          mae.objective(ix, visible: (0...16).to_a.sample(8, random: rng).sort)
        end
        c.results[:mae] = c.evaluate(mae) { { masked_patch_mse: mae.objective(iv, visible: visible).item, visible: visible } }
        c.artifacts.json("mae-reconstruction", c.evaluate(mae) { MaskedAutoencoder.unpatchify(mae.call(iv, visible: visible)).cpu.to_a })
        c.artifacts.image_grid("mae-images", { target: valid_images, reconstruction: c.evaluate(mae) { MaskedAutoencoder.unpatchify(mae.call(iv, visible: visible)).cpu.to_a } })
        c.save("mae-model", mae)
      end
    end
  end
end
