module EasyAILearning
  module Autoencoder
    class Convolutional < Torch::NN::Module
      def initialize
        super()
        @encoder = Torch::NN::Conv2d.new(1, 4, 3, padding: 1)
        @decoder = Torch::NN::Conv2d.new(4, 1, 3, padding: 1)
      end

      def encode(x)
        Torch::NN::Functional.avg_pool2d(Torch.relu(@encoder.call(x)), 2)
      end

      def forward(x)
        z = encode(x)
        enlarged = Torch::NN::Functional.interpolate(z, size: x.shape[-2..], mode: "nearest")
        Torch.sigmoid(@decoder.call(enlarged))
      end
    end
  end
end
