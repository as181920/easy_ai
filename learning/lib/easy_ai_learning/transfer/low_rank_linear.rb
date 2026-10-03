module EasyAILearning
  module Transfer
    class LowRankLinear < Torch::NN::Module
      attr_reader :base, :a, :b, :scale

      def initialize(base, rank: 2, alpha: 2.0)
        super()
        raise ArgumentError, "Positive rank required" unless rank > 0
        @base, @scale = base, alpha / rank
        base.parameters.each { |p| p.requires_grad = false }
        @a = Torch::NN::Linear.new(base.weight.shape[1], rank, bias: false)
        @b = Torch::NN::Linear.new(rank, base.weight.shape[0], bias: false)
        Torch.no_grad { b.weight.zero! }
      end

      def forward(x)
        base.call(x) + scale * b.call(a.call(x))
      end

      def merged_weight
        base.weight.detach + scale * Torch.matmul(b.weight.detach, a.weight.detach)
      end
    end
  end
end
