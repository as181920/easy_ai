module EasyAILearning
  module Transformer
    class SequenceModel < Torch::NN::Module
      def initialize(vocab: 7, width: 8, max_length: 8, positions: true, residual: true, normalize: true, post_norm: false)
        super()
        @positions, @max_length = positions, max_length
        @embedding = Torch::NN::Embedding.new(vocab, width)
        @position = positions ? Torch::NN::Embedding.new(max_length, width) : nil
        @block = EncoderBlock.new(width: width, residual: residual, normalize: normalize, post_norm: post_norm)
        @head = Torch::NN::Linear.new(width, vocab)
      end

      def forward(tokens, padding_mask: nil)
        raise ArgumentError, "Sequence too long" unless tokens.shape[1] <= @max_length
        x = @embedding.call(tokens)
        if @positions
          positions = Torch.arange(tokens.shape[1], dtype: :int64, device: tokens.device)
          x = x + @position.call(positions).unsqueeze(0)
        end
        @head.call(@block.call(x, padding_mask: padding_mask))
      end
    end
  end
end
