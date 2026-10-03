module EasyAILearning
  module Foundations
    class Linear
      attr_reader :weights, :bias, :history

      def initialize(features:, logistic: false)
        @weights, @bias, @history, @logistic = Array.new(features, 0.0), 0.0, [], logistic
      end

      def predict(row)
        score = Math.dot(weights, row) + bias
        @logistic ? Math.sigmoid(score) : score
      end

      def gradients(rows, targets)
        residual = rows.zip(targets).map { |row, y| predict(row) - y }
        multiplier = @logistic ? 1.0 : 2.0
        { weights: weights.each_index.map { |i| multiplier * rows.each_index.sum { |j| residual[j] * rows[j][i] } / rows.size },
          bias: multiplier * residual.sum / rows.size }
      end

      def loss(rows, targets)
        if @logistic
          rows.zip(targets).sum do |row, y|
            z = Math.dot(weights, row) + bias
            [z, 0].max - y * z + ::Math.log(1 + ::Math.exp(-z.abs))
          end / rows.size
        else
          Math.mse(rows.map { |row| predict(row) }, targets)
        end
      end

      def fit(rows, targets, steps: 100, lr: 0.1)
        steps.times do
          g = gradients(rows, targets)
          @weights = weights.zip(g[:weights]).map { |v, grad| v - lr * grad }
          @bias -= lr * g[:bias]
          history << loss(rows, targets)
        end
        self
      end
    end
  end
end
