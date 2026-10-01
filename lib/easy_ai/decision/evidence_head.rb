module EasyAI
  module Decision
    # Question-conditioned sentence scoring; annotations never enter forward inputs.
    class EvidenceHead < Torch::NN::Module
      def initialize(hidden_size)
        super()
        @query = Torch::NN::Linear.new(hidden_size, hidden_size, bias: false)
        @scale = Math.sqrt(hidden_size)
      end

      def forward(query, memory:, sentence_mask:, candidate_mask:)
        b, k = candidate_mask.shape
        weights = candidate_mask.to(dtype: :float32).unsqueeze(-1)
        pooled = (query.view([b, k, -1]) * weights).sum(dim: 1) / weights.sum(dim: 1).clamp(min: 1.0)
        sentences = sentence_mask.to(dtype: :float32)
        counts = sentences.sum(dim: -1)
        states = Torch.matmul(sentences, memory) / counts.clamp(min: 1.0).unsqueeze(-1)
        scores = (states * @query.call(pooled).unsqueeze(1)).sum(dim: -1) / @scale
        scores.masked_fill(counts.eq(0), -Float::INFINITY)
      end
    end
  end
end
