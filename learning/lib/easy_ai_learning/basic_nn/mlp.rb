module EasyAILearning
  module BasicNN
    class Mlp < Torch::NN::Module
      attr_reader :hidden, :output, :dropout_probability, :last_hidden

      def initialize(input: 2, hidden: 12, output: 2, dropout: 0.0, activation: :relu)
        super()
        activation = activation.to_s.to_sym
        raise ArgumentError, "Unknown activation" unless %i[relu tanh].include?(activation)
        @hidden = Torch::NN::Linear.new(input, hidden)
        @output = Torch::NN::Linear.new(hidden, output)
        @dropout = Torch::NN::Dropout.new(p: dropout)
        @dropout_probability, @activation = dropout, activation
      end

      def encode(x)
        z = hidden.call(x)
        @last_hidden = @activation == :tanh ? Torch.tanh(z) : Torch.relu(z)
        @dropout.call(@last_hidden)
      end

      def forward(x)
        output.call(encode(x))
      end
    end
  end
end
