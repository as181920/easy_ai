module EasyAILearning
  module Training
    module Math
      module_function

      def masked_cross_entropy(logits, targets, mask: nil)
        if mask
          raise ArgumentError, "Mask/target shape mismatch" unless mask.shape == targets.shape
          raise ArgumentError, "No valid targets" unless mask.to(dtype: :float32).sum.item > 0
          targets = targets.masked_fill(mask.to(dtype: :bool).logical_not, 0)
        end
        count = logits.shape[-1]
        loss = Torch::NN::Functional.cross_entropy(logits.reshape([-1, count]), targets.reshape([-1]), reduction: "none")
        return loss.mean unless mask
        weights = mask.reshape([-1]).to(dtype: logits.dtype)
        (loss * weights).sum / weights.sum
      end

      def clipped_gradients(parameters, max_norm)
        raise ArgumentError, "max_norm must be positive" unless max_norm > 0

        norms = parameters.filter_map { |p| p.grad&.norm&.item }
        before = ::Math.sqrt(norms.sum { |v| v * v })
        raise FloatDomainError, "Nonfinite gradient norm" unless before.finite?
        factor = [1.0, max_norm / (before + 1e-12)].min
        Torch.no_grad { parameters.each { |p| p.grad.mul!(factor) if p.grad } }
        { before: before, after: before * factor }
      end

      def learning_rate(step, total:, base:, warmup: 0)
        raise ArgumentError, "Invalid schedule" unless total > 0 && step >= 0 && warmup >= 0 && warmup < total && base > 0
        return base * (step + 1).to_f / warmup if step < warmup

        progress = [[(step - warmup).to_f / [total - warmup - 1, 1].max, 0.0].max, 1.0].min
        base * 0.5 * (1 + ::Math.cos(::Math::PI * progress))
      end

      def dropout(x, probability:, mask: nil, training: true)
        raise ArgumentError, "Invalid dropout" unless probability >= 0 && probability < 1
        return x unless training && probability > 0

        mask ||= Torch.rand_like(x).ge(probability)
        x * mask.to(dtype: x.dtype) / (1 - probability)
      end

      def standardize_fit(rows)
        raise ArgumentError, "Empty data" if rows.empty?
        columns = rows.transpose
        mean = columns.map { |c| c.sum.to_f / c.size }
        std = columns.each_with_index.map do |c, i|
          [::Math.sqrt(c.sum { |v| (v - mean[i])**2 } / c.size), 1e-8].max
        end
        { mean: mean, std: std }
      end

      def standardize(rows, stats)
        rows.map { |row| row.each_with_index.map { |v, i| (v - stats.fetch(:mean)[i]) / stats.fetch(:std)[i] } }
      end
    end
  end
end
