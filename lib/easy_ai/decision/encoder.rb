module EasyAI
  module Decision
    class Encoder < Torch::NN::Module
      attr_reader :embedding, :blocks

      def initialize(config)
        super()
        c = config[:model]
        @hidden_size = c["hidden_size"]
        @position_scale = c["position_scale"]
        @rotary = c["position_encoding"] == "rotary"
        @embedding = Torch::NN::Embedding.new(c["vocab_size"], @hidden_size, padding_idx: 0)
        @embedding_norm = Torch::NN::LayerNorm.new(@hidden_size) if c["embedding_norm"]
        widths = c["ffn_sizes"] || Array.new(c["encoder_layers"], c["ffn_size"])
        @blocks = Torch::NN::ModuleList.new(widths.map do |width|
          NN::EncoderBlock.new(hidden_size: @hidden_size, heads: c["attention_heads"], intermediate_size: width, dropout: c["dropout"], rotary: @rotary)
        end)
        @norm = Torch::NN::LayerNorm.new(@hidden_size)
        @dropout = Torch::NN::Dropout.new(p: c["dropout"])
      end

      def forward(ids, mask:)
        x = @embedding.call(ids)
        unless @rotary
          # Legacy additive sinusoidal positions remain checkpoint-compatible.
          positions = Torch.arange(ids.shape[1], dtype: :float32, device: ids.device).unsqueeze(1)
          frequencies = Torch.exp(Torch.arange(0, @hidden_size, 2, dtype: :float32, device: ids.device) * (-Math.log(10_000.0) / @hidden_size))
          angles = positions * frequencies.unsqueeze(0)
          positional = Torch.stack([Torch.sin(angles), Torch.cos(angles)], dim: -1).view([ids.shape[1], @hidden_size])
          x = x + positional.unsqueeze(0) * @position_scale
        end
        x = @embedding_norm.call(x) if @embedding_norm
        x = @dropout.call(x)
        @blocks.each { |block| x = block.call(x, mask: mask) }
        @norm.call(x)
      end
    end
  end
end
