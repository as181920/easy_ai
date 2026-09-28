module EasyAI
  module Distillation
    module Losses
      module SoftTargets
        # Each row contains only real candidates; supports varying candidate counts.
        # Probabilities are stored at T=1 and re-tempered here, never token-aligned.
        def self.call(logits, distributions, temperature: 1.0)
          unless temperature.is_a?(Numeric) && temperature.finite? && temperature > 0
            raise ArgumentError, "Temperature must be finite and positive"
          end
          unless logits.shape.length == 2 && logits.shape[0] == distributions.length && distributions.any?
            raise ArgumentError, "Expected one distribution per logit row"
          end
          losses = distributions.each_with_index.map do |values, index|
            unless values.is_a?(Array) && values.size.between?(2, logits.shape[1]) &&
                values.all? { |value| value.is_a?(Numeric) && value.finite? && value >= 0 } && (values.sum - 1.0).abs < 1e-6
              raise ArgumentError, "Expected normalized, complete candidate probabilities at T=1"
            end
            # Re-temper in log space; preserve exact zeros, avoid 0 * log(0).
            logs = values.map { |value| value.zero? ? -Float::INFINITY : Math.log(value) / temperature }
            maximum = logs.max
            weights = logs.map { |value| Math.exp(value - maximum) }
            total = weights.sum
            probabilities = weights.map { |value| value / total }
            target = Torch.tensor(probabilities, dtype: logits.dtype, device: logits.device)
            target_logs = Torch.tensor(probabilities.map { |value| value.zero? ? 0.0 : Math.log(value) }, dtype: logits.dtype, device: logits.device)
            student_logs = Torch::NN::Functional.log_softmax(logits[index][0...values.size] / temperature, -1)
            (target * (target_logs - student_logs)).sum * temperature**2
          end
          Torch.stack(losses).mean
        end
      end
    end
  end
end
