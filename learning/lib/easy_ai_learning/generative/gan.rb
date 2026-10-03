module EasyAILearning
  module Generative
    class Gan < Torch::NN::Module
      attr_reader :generator, :discriminator

      def initialize
        super()
        @generator = BasicNN::Mlp.new(input: 2, hidden: 16, output: 2, activation: :tanh)
        @discriminator = BasicNN::Mlp.new(input: 2, hidden: 16, output: 1)
      end

      def forward(noise)
        generator.call(noise)
      end

      def discriminator_loss(real, fake)
        real_logits, fake_logits = discriminator.call(real), discriminator.call(fake.detach)
        Torch::NN::Functional.binary_cross_entropy_with_logits(real_logits, Torch.ones_like(real_logits)) +
          Torch::NN::Functional.binary_cross_entropy_with_logits(fake_logits, Torch.zeros_like(fake_logits))
      end

      def generator_loss(noise)
        logits = discriminator.call(generator.call(noise))
        Torch::NN::Functional.binary_cross_entropy_with_logits(logits, Torch.ones_like(logits))
      end
    end
  end
end
