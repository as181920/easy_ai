module EasyAI
  module Decision
    class InteractionBlock < Torch::NN::Module
      def initialize(config)
        super()
        c = config[:model]
        @norm1 = Torch::NN::LayerNorm.new(c["hidden_size"])
        @norm2 = Torch::NN::LayerNorm.new(c["hidden_size"])
        @attention = NN::Attention.new(hidden_size: c["hidden_size"], heads: c["attention_heads"], dropout: c["dropout"])
        @ffn = NN::FeedForward.new(hidden_size: c["hidden_size"], intermediate_size: c["ffn_size"], dropout: c["dropout"])
        @dropout = Torch::NN::Dropout.new(p: c["dropout"])
      end

      def forward(query, memory:, memory_mask:)
        x = query + @dropout.call(@attention.call(@norm1.call(query), memory: memory, key_mask: memory_mask))
        x + @dropout.call(@ffn.call(@norm2.call(x)))
      end
    end
  end
end
