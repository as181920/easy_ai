module EasyAILearning
  module Generative
    class MaskedAutoencoder < Torch::NN::Module
      def initialize(width: 8)
        super()
        @projection = Torch::NN::Linear.new(4, width)
        @position = Torch::NN::Embedding.new(16, width)
        @encoder = Transformer::EncoderBlock.new(width: width)
        @decoder = Transformer::EncoderBlock.new(width: width)
        @mask_token = Torch::NN::Parameter.new(Torch.zeros([width]))
        @head = Torch::NN::Linear.new(width, 4)
      end

      def self.patchify(images)
        raise ArgumentError, "Expected [batch,1,8,8]" unless images.shape[1..] == [1, 8, 8]
        images.reshape([images.shape[0], 1, 4, 2, 4, 2]).permute([0, 2, 4, 1, 3, 5]).contiguous.reshape([images.shape[0], 16, 4])
      end

      def self.unpatchify(patches)
        patches.reshape([patches.shape[0], 4, 4, 1, 2, 2]).permute([0, 3, 1, 4, 2, 5]).contiguous.reshape([patches.shape[0], 1, 8, 8])
      end

      def forward(images, visible:)
        raise ArgumentError, "Invalid visible patches" unless !visible.empty? && visible.uniq == visible && visible.all? { |i| i.between?(0, 15) }
        patches = self.class.patchify(images)
        positions = @position.call(Torch.arange(16, dtype: :int64, device: images.device)).unsqueeze(0)
        indices = Torch.tensor(visible, dtype: :int64, device: images.device)
        encoded = @encoder.call((@projection.call(patches) + positions).index_select(1, indices))
        restored = 16.times.map do |i|
          index = visible.index(i)
          index ? encoded.narrow(1, index, 1).squeeze(1) : @mask_token.unsqueeze(0).expand(images.shape[0], -1)
        end
        @head.call(@decoder.call(Torch.stack(restored, dim: 1) + positions))
      end

      def objective(images, visible:)
        missing = (0...16).to_a - visible
        raise ArgumentError, "Need masked patches" if missing.empty?
        indices = Torch.tensor(missing, dtype: :int64, device: images.device)
        prediction = forward(images, visible: visible).index_select(1, indices)
        Torch::NN::Functional.mse_loss(prediction, self.class.patchify(images).index_select(1, indices))
      end
    end
  end
end
