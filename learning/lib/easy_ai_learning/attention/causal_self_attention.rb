module EasyAILearning
  module Attention
    class CausalSelfAttention < MultiHead
      def forward(x, padding_mask: nil)
        super(x, causal: true, padding_mask: padding_mask)
      end
    end
  end
end
