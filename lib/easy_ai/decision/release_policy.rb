module EasyAI
  module Decision
    class ReleasePolicy
      REQUIREMENTS = {
        "count" => 300,
        "accuracy" => 0.80,
        "balanced_accuracy" => 0.70,
        "coverage" => 0.30,
        "accepted_count" => 100,
        "accepted_accuracy" => 0.90,
        "accepted_accuracy_lower_95" => 0.85
      }.freeze

      def self.fit(probabilities, targets, minimum_accuracy: REQUIREMENTS.fetch("accepted_accuracy"),
                   minimum_lower_bound: REQUIREMENTS.fetch("accepted_accuracy_lower_95"))
        unless (REQUIREMENTS.fetch("accepted_accuracy")..1.0).cover?(minimum_accuracy) &&
            (REQUIREMENTS.fetch("accepted_accuracy_lower_95")..1.0).cover?(minimum_lower_bound)
          raise ArgumentError, "Calibration guards cannot weaken release requirements"
        end
        candidates = (0..100).map do |index|
          ReleaseMetrics.measure(probabilities, targets, threshold: index / 100.0)
        end
        eligible = candidates.select do |metrics|
          %w[coverage accepted_count accepted_accuracy accepted_accuracy_lower_95].all? do |key|
            minimum = { "accepted_accuracy" => minimum_accuracy, "accepted_accuracy_lower_95" => minimum_lower_bound }.fetch(key, REQUIREMENTS.fetch(key))
            metrics[key] && metrics[key] >= minimum
          end
        end
        best = eligible.max_by { |metrics| [metrics.fetch("coverage"), -metrics.fetch("threshold")] }
        { "available" => !best.nil?, "threshold" => best&.fetch("threshold"), "calibration" => best }
      end

      def self.failures(metrics, expected_labels:)
        failures = REQUIREMENTS.filter_map do |key, minimum|
          "#{key}: #{metrics[key].inspect} < #{minimum}" unless metrics[key] && metrics[key] >= minimum
        end
        absent = expected_labels.map(&:to_s) - metrics.fetch("recall").keys
        failures << "Missing target labels: #{absent.join(', ')}" unless absent.empty?
        failures
      end
    end
  end
end
