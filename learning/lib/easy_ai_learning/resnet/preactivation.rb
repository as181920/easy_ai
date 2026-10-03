module EasyAILearning
  module Resnet
    class Preactivation < Torch::NN::Module
      def initialize(channels: 4)
        super()
        @norm1, @norm2 = Torch::NN::BatchNorm2d.new(channels), Torch::NN::BatchNorm2d.new(channels)
        @conv1 = Torch::NN::Conv2d.new(channels, channels, 3, padding: 1, bias: false)
        @conv2 = Torch::NN::Conv2d.new(channels, channels, 3, padding: 1, bias: false)
      end

      def forward(x)
        residual = @conv1.call(Torch.relu(@norm1.call(x)))
        x + @conv2.call(Torch.relu(@norm2.call(residual)))
      end
    end
  end
end
