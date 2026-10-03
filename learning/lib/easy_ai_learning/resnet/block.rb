module EasyAILearning
  module Resnet
    class Block < Torch::NN::Module
      attr_reader :conv1, :conv2, :shortcut

      def initialize(input: 4, output: 4, stride: 1, normalize: true)
        super()
        @conv1 = Torch::NN::Conv2d.new(input, output, 3, stride: stride, padding: 1, bias: !normalize)
        @conv2 = Torch::NN::Conv2d.new(output, output, 3, padding: 1, bias: !normalize)
        @norm1 = normalize ? Torch::NN::BatchNorm2d.new(output) : Torch::NN::Identity.new
        @norm2 = normalize ? Torch::NN::BatchNorm2d.new(output) : Torch::NN::Identity.new
        @shortcut = input == output && stride == 1 ? Torch::NN::Identity.new : Torch::NN::Conv2d.new(input, output, 1, stride: stride, bias: false)
      end

      def residual(x)
        @norm2.call(conv2.call(Torch.relu(@norm1.call(conv1.call(x)))))
      end

      def forward(x)
        Torch.relu(shortcut.call(x) + residual(x))
      end
    end
  end
end
