module EasyAILearning
  module Transfer
    class AdaptedMlp < BasicNN::Mlp
      def initialize(input: 2, hidden: 12, output: 2, rank: 2)
        super(input: input, hidden: hidden, output: output)
        self.hidden.parameters.each { |p| p.requires_grad = false }
        @output = LowRankLinear.new(@output, rank: rank)
      end
    end
  end
end
