module EasyAILearning
  module Resnet
    class Model < Torch::NN::Module
      def initialize(depth: 3, channels: 4, residual: true, normalize: true, classes: 2)
        super()
        @residual = residual
        @stem = Torch::NN::Conv2d.new(1, channels, 3, padding: 1)
        @blocks = Torch::NN::ModuleList.new(Array.new(depth) { Block.new(input: channels, output: channels, normalize: normalize) })
        @head = Torch::NN::Linear.new(channels, classes)
      end

      def features(x)
        x = Torch.relu(@stem.call(x))
        @blocks.each { |block| x = @residual ? block.call(x) : Torch.relu(block.residual(x)) }
        x.mean([2, 3])
      end

      def forward(x)
        @head.call(features(x))
      end
    end
  end
end
