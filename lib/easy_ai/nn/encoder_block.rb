module EasyAI
  module NN
    class EncoderBlock < Torch::NN::Module
      attr_reader :attention, :ffn

      def initialize(hidden_size:, heads:, intermediate_size:, dropout: 0.0, rotary: false)
        super()
        @norm1 = Torch::NN::LayerNorm.new(hidden_size)
        @norm2 = Torch::NN::LayerNorm.new(hidden_size)
        @attention = Attention.new(hidden_size: hidden_size, heads: heads, dropout: dropout, rotary: rotary)
        @ffn = FeedForward.new(hidden_size: hidden_size, intermediate_size: intermediate_size, dropout: dropout)
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      def forward(x, mask:)
        normalized = @norm1.call(x)
        x = x + @dropout.call(@attention.call(normalized, key_mask: mask))
        x + @dropout.call(@ffn.call(@norm2.call(x)))
      end

      def identity!
        Torch.no_grad do
          [attention.output, ffn.down].each do |projection|
            projection.weight.zero!
            projection.bias.zero!
          end
        end
        self
      end
    end
  end
end
