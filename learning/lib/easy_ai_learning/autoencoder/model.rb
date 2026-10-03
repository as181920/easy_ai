module EasyAILearning
  module Autoencoder
    class Model < Torch::NN::Module
      attr_reader :encoder, :decoder

      def initialize(input: 4, latent: 2, nonlinear: true)
        super()
        @nonlinear = nonlinear
        @encoder = nonlinear ? BasicNN::Mlp.new(input: input, hidden: 12, output: latent, activation: :tanh) : Torch::NN::Linear.new(input, latent)
        @decoder = nonlinear ? BasicNN::Mlp.new(input: latent, hidden: 12, output: input, activation: :tanh) : Torch::NN::Linear.new(latent, input)
      end

      def encode(x)
        encoder.call(x)
      end

      def forward(x)
        decoder.call(encode(x))
      end

      def objective(input, target, sparsity: 0.0)
        latent = encode(input)
        reconstruction = decoder.call(latent)
        Torch::NN::Functional.mse_loss(reconstruction, target) + sparsity * latent.abs.mean
      end

      def self.corrupt(x, mask:)
        raise ArgumentError, "Mask shape mismatch" unless x.shape == mask.shape
        x * mask.to(dtype: x.dtype)
      end
    end
  end
end
