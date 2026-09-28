module EasyAI
  module Decision
    class Calibrator
      attr_reader :temperature

      def initialize(temperature: 1.0)
        raise ArgumentError, "Temperature must be positive and finite" unless temperature.is_a?(Numeric) && temperature.finite? && temperature > 0
        @temperature = temperature.to_f
      end

      def probabilities(logits)
        raise ArgumentError, "Need two or more finite scores" unless logits.length >= 2 && logits.all?(&:finite?)
        maximum = logits.max
        shifted = logits.map { |score| (score - maximum) / temperature }
        exponentials = shifted.map { |score| Math.exp(score) }
        total = exponentials.sum
        exponentials.map { |value| value / total }
      end

      def fit(logits, targets)
        raise ArgumentError, "Empty or mismatched calibration set" if logits.empty? || logits.length != targets.length
        # Deterministic bounded optimization of log(T); T=1 is also a candidate.
        left, right = Math.log(0.05), Math.log(20.0)
        80.times do
          a, b = left + (right - left) / 3, right - (right - left) / 3
          if nll(logits, targets, Math.exp(a)) <= nll(logits, targets, Math.exp(b))
            right = b
          else
            left = a
          end
        end
        candidate = Math.exp((left + right) / 2)
        @temperature = nll(logits, targets, candidate) < nll(logits, targets, 1.0) ? candidate : 1.0
        self
      end

      def nll(logits, targets, temperature = @temperature)
        logits.zip(targets).sum do |row, target|
          raise ArgumentError, "Invalid target or logits" unless target.is_a?(Integer) && target.between?(0, row.length - 1) && row.all?(&:finite?)
          maximum = row.max
          scaled = row.map { |value| (value - maximum) / temperature }
          Math.log(scaled.sum { |value| Math.exp(value) }) - scaled.fetch(target)
        end / logits.length
      end
    end
  end
end
