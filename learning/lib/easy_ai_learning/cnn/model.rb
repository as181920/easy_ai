module EasyAILearning
  module CNN
    class Model < Torch::NN::Module
      attr_reader :conv1, :conv2, :last_features

      def initialize(classes: 2, channels: 4, normalize: false)
        super()
        @conv1 = Torch::NN::Conv2d.new(1, channels, 3, padding: 1)
        @conv2 = Torch::NN::Conv2d.new(channels, channels * 2, 3, padding: 1)
        @norm1 = normalize ? Torch::NN::BatchNorm2d.new(channels) : Torch::NN::Identity.new
        @norm2 = normalize ? Torch::NN::BatchNorm2d.new(channels * 2) : Torch::NN::Identity.new
        @head = Torch::NN::Linear.new(channels * 2, classes)
      end

      def features(x)
        x = Torch.relu(@norm1.call(conv1.call(x)))
        x = Torch::NN::Functional.max_pool2d(x, 2)
        @last_features = Torch.relu(@norm2.call(conv2.call(x)))
        @last_features.mean([2, 3])
      end

      def forward(x)
        @head.call(features(x))
      end
    end
  end
end
