module EasyAILearning
  module Generative
    class Vae < Torch::NN::Module
      attr_reader :encoder, :decoder, :latent_size

      def initialize(input: 4, latent: 2)
        super()
        @latent_size = latent
        @encoder = BasicNN::Mlp.new(input: input, hidden: 12, output: latent * 2, activation: :tanh)
        @decoder = BasicNN::Mlp.new(input: latent, hidden: 12, output: input, activation: :tanh)
      end

      def distribution(x)
        encoded = encoder.call(x)
        [encoded.narrow(1, 0, latent_size), encoded.narrow(1, latent_size, latent_size)]
      end

      def self.reparameterize(mean, log_variance, noise: nil)
        mean + (0.5 * log_variance).exp * (noise || Torch.randn_like(mean))
      end

      def self.kl(mean, log_variance)
        -0.5 * (1 + log_variance - mean.square - log_variance.exp).sum(1).mean
      end

      def forward(x)
        mean, = distribution(x)
        decoder.call(mean) # deterministic inference reconstruction
      end

      def objective(x, beta: 1.0, noise: nil)
        mean, log_variance = distribution(x)
        z = self.class.reparameterize(mean, log_variance, noise: noise)
        reconstruction = (decoder.call(z) - x).square.sum(1).mean
        reconstruction + beta * self.class.kl(mean, log_variance)
      end
    end
  end
end
