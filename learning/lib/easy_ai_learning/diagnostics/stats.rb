module EasyAILearning
  module Diagnostics
    module Stats
      module_function

      def summarize(values, threshold: 1e-3)
        values = values.flatten.map(&:to_f)
        finite = values.select(&:finite?).sort
        result = { count: values.size, nonfinite: values.size - finite.size }
        return result if finite.empty?

        mean = finite.sum / finite.size
        rms = ::Math.sqrt(finite.sum { |v| v * v } / finite.size)
        result.merge(mean: mean, std: ::Math.sqrt(finite.sum { |v| (v - mean)**2 } / finite.size), rms: rms,
          min: finite.first, max: finite.last, max_abs: finite.map(&:abs).max,
          q05: quantile(finite, 0.05), median: quantile(finite, 0.5), q95: quantile(finite, 0.95),
          near_zero_fraction: finite.count { |v| v.abs < threshold }.to_f / finite.size,
          relative_near_zero_fraction: finite.count { |v| v.abs < rms * 0.01 }.to_f / finite.size)
      end

      def quantile(sorted, fraction)
        position = fraction * (sorted.size - 1)
        lo, hi = position.floor, position.ceil
        sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo)
      end

      def histogram(values, bins: 12)
        raise ArgumentError, "Positive bins required" unless bins > 0
        values = values.flatten.select(&:finite?)
        return [] if values.empty?
        lo, hi = values.minmax
        hi = lo + 1.0 if lo == hi
        width, counts = (hi - lo).to_f / bins, Array.new(bins, 0)
        values.each { |v| counts[[((v - lo) / width).floor, bins - 1].min] += 1 }
        counts.each_with_index.map { |count, i| [lo + (i + 0.5) * width, count] }
      end

      def spectrum(matrix)
        raise ArgumentError, "Matrix required" unless matrix.dim == 2
        singular = Torch.svd(matrix.detach)[1].cpu.to_a
        total = singular.sum
        probabilities = total.zero? ? [] : singular.map { |v| v / total }.select { |v| v > 0 }
        entropy = -probabilities.sum { |p| p * ::Math.log(p) }
        { singular_values: singular, effective_rank: total.zero? ? 0 : ::Math.exp(entropy) }
      end

      def model(model)
        model.named_parameters.to_h do |name, parameter|
          [name, { shape: parameter.shape, weights: summarize(parameter.detach.cpu.to_a),
            gradient: parameter.grad ? summarize(parameter.grad.detach.cpu.to_a) : { missing: true } }]
        end
      end

      def snapshot(model)
        model.named_parameters.transform_values { |p| p.detach.clone }
      end

      def updates(model, before)
        model.named_parameters.to_h do |name, p|
          absolute = (p.detach - before.fetch(name)).norm.item
          [name, { absolute: absolute, relative: absolute / (before.fetch(name).norm.item + 1e-12) }]
        end
      end

      def activation(tensor)
        values = tensor.detach.cpu.to_a.flatten
        summarize(values).merge(zero_fraction: values.count(&:zero?).to_f / values.size)
      end
    end
  end
end
