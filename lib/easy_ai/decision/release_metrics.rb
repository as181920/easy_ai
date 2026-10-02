module EasyAI
  module Decision
    # Acceptance statistics for a frozen confidence threshold, never a semantic rule.
    class ReleaseMetrics
      def self.measure(probabilities, targets, threshold:)
        validate!(probabilities, targets, threshold)
        predictions = probabilities.map { |row| row.each_index.max_by { |index| row[index] } }
        accepted = probabilities.each_index.select { |index| probabilities[index].max >= threshold }
        correct = predictions.each_index.count { |index| predictions[index] == targets[index] }
        accepted_correct = accepted.count { |index| predictions[index] == targets[index] }
        recalls = targets.uniq.to_h do |label|
          indexes = targets.each_index.select { |index| targets[index] == label }
          [label.to_s, indexes.count { |index| predictions[index] == label }.fdiv(indexes.size)]
        end
        { "count" => targets.size, "accuracy" => correct.fdiv(targets.size),
          "balanced_accuracy" => recalls.values.sum.fdiv(recalls.size), "recall" => recalls,
          "threshold" => threshold, "accepted_count" => accepted.size, "coverage" => accepted.size.fdiv(targets.size),
          "accepted_accuracy" => accepted.empty? ? nil : accepted_correct.fdiv(accepted.size),
          "accepted_accuracy_lower_95" => wilson_lower(accepted_correct, accepted.size) }
      end

      def self.wilson_lower(correct, count)
        return nil if count.zero?
        z = 1.959963984540054
        proportion = correct.fdiv(count)
        (proportion + z**2 / (2 * count) - z * Math.sqrt((proportion * (1 - proportion) + z**2 / (4 * count)) / count)) /
          (1 + z**2 / count)
      end

      def self.validate!(probabilities, targets, threshold)
        unless threshold.is_a?(Numeric) && threshold.finite? && (0.0..1.0).cover?(threshold)
          raise ArgumentError, "Confidence threshold must be between zero and one"
        end
        raise ArgumentError, "Empty or mismatched release panel" if targets.empty? || probabilities.size != targets.size
        probabilities.zip(targets).each do |row, target|
          unless row.is_a?(Array) && row.size >= 2 && row.all? { |value| value.is_a?(Numeric) && value.finite? && (0.0..1.0).cover?(value) } &&
              (row.sum - 1.0).abs <= 1e-6 && target.is_a?(Integer) && target.between?(0, row.size - 1)
            raise ArgumentError, "Invalid release probability vector or target"
          end
        end
      end
      private_class_method :validate!
    end
  end
end
