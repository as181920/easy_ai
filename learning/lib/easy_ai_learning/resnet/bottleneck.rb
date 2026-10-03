module EasyAILearning
  module Resnet
    class Bottleneck < Torch::NN::Module
      attr_reader :shortcut

      def initialize(input: 4, output: 8, inner: 2, stride: 1)
        super()
        @reduce = Torch::NN::Conv2d.new(input, inner, 1, bias: false)
        @spatial = Torch::NN::Conv2d.new(inner, inner, 3, padding: 1, stride: stride, bias: false)
        @expand = Torch::NN::Conv2d.new(inner, output, 1, bias: false)
        @norm1, @norm2, @norm3 = Torch::NN::BatchNorm2d.new(inner), Torch::NN::BatchNorm2d.new(inner), Torch::NN::BatchNorm2d.new(output)
        @shortcut = input == output && stride == 1 ? Torch::NN::Identity.new : Torch::NN::Conv2d.new(input, output, 1, stride: stride, bias: false)
      end

      def forward(x)
        residual = Torch.relu(@norm1.call(@reduce.call(x)))
        residual = Torch.relu(@norm2.call(@spatial.call(residual)))
        Torch.relu(shortcut.call(x) + @norm3.call(@expand.call(residual)))
      end
    end
  end
end
