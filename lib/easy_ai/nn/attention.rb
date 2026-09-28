module EasyAI
  module NN
    # Explicit scaled dot-product attention. True in key_mask means a real token.
    class Attention < Torch::NN::Module
      attr_reader :output

      def initialize(hidden_size:, heads:, dropout: 0.0, rotary: false)
        super()
        raise ArgumentError, "hidden size/head mismatch" unless (hidden_size % heads).zero?
        @hidden, @heads, @head_dim = hidden_size, heads, hidden_size / heads
        @rotary = rotary
        @query = Torch::NN::Linear.new(hidden_size, hidden_size)
        @key = Torch::NN::Linear.new(hidden_size, hidden_size)
        @value = Torch::NN::Linear.new(hidden_size, hidden_size)
        @output = Torch::NN::Linear.new(hidden_size, hidden_size)
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      # Shapes: query [B,Q,D], memory [B,L,D], key_mask [B,L].
      def forward(query, memory: query, key_mask:)
        q = split(@query.call(query))
        k = split(@key.call(memory))
        v = split(@value.call(memory))
        q, k = RotaryPosition.call(q), RotaryPosition.call(k) if @rotary
        scores = Torch.matmul(q, k.transpose(-2, -1)) / Math.sqrt(@head_dim)
        scores = scores.masked_fill(Torch.logical_not(key_mask.unsqueeze(1).unsqueeze(1)), -Float::INFINITY)
        probabilities = Torch::NN::Functional.softmax(scores, dim: -1)
        values = Torch.matmul(@dropout.call(probabilities), v)
        @output.call(values.transpose(1, 2).contiguous.view([query.shape[0], query.shape[1], @hidden]))
      end

      private

      def split(tensor)
        tensor.view([tensor.shape[0], tensor.shape[1], @heads, @head_dim]).transpose(1, 2)
      end
    end
  end
end
