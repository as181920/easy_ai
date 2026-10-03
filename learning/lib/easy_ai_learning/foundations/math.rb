module EasyAILearning
  module Foundations
    module Math
      module_function

      def dot(a, b)
        raise ArgumentError, "Different dimensions" unless a.size == b.size
        a.zip(b).sum { |x, y| x * y }
      end

      def softmax(logits)
        raise ArgumentError, "Empty logits" if logits.empty?
        exp = logits.map { |v| ::Math.exp(v - logits.max) }
        exp.map { |v| v / exp.sum }
      end

      def sigmoid(value)
        value >= 0 ? 1.0 / (1 + ::Math.exp(-value)) : ::Math.exp(value) / (1 + ::Math.exp(value))
      end

      def cross_entropy(logits, target)
        raise ArgumentError, "Target out of range" unless target.between?(0, logits.size - 1)
        largest = logits.max
        largest + ::Math.log(logits.sum { |v| ::Math.exp(v - largest) }) - logits[target]
      end

      def mse(prediction, target)
        raise ArgumentError, "Invalid samples" unless prediction.size == target.size && !target.empty?
        prediction.zip(target).sum { |a, b| (a - b)**2 }.to_f / target.size
      end

      def confusion(predictions, targets, classes: 2)
        raise ArgumentError, "Invalid samples" unless predictions.size == targets.size && !targets.empty?
        matrix = Array.new(classes) { Array.new(classes, 0) }
        predictions.zip(targets).each { |p, t| matrix[t][p] += 1 }
        accuracy = classes.times.sum { |i| matrix[i][i] }.to_f / targets.size
        { matrix: matrix, accuracy: accuracy }
      end

      def finite_difference(value, epsilon: 1e-5, &function)
        (function.call(value + epsilon) - function.call(value - epsilon)) / (2 * epsilon)
      end
    end
  end
end
