module EasyAILearning
  module Foundations
    class Ensemble
      attr_reader :trees, :base

      def fit(rows, targets, kind: :forest, count: 8, depth: 2, lr: 0.2, seed: 1337)
        kind = kind.to_s.to_sym
        raise ArgumentError, "Invalid ensemble kind" unless %i[forest boosting].include?(kind)
        @kind, @lr, @trees = kind, lr, []
        @base = kind == :boosting ? targets.sum.to_f / targets.size : 0.0
        rng = Random.new(seed)
        count.times do
          if kind == :forest
            indices = Array.new(rows.size) { rng.rand(rows.size) }
            trees << Tree.new.fit(indices.map { |i| rows[i] }, indices.map { |i| targets[i] }, depth: depth, max_features: [1, ::Math.sqrt(rows.first.size).ceil].max, seed: rng.rand(1000000))
          else
            residual = rows.zip(targets).map { |r, y| y - predict(r) }
            trees << Tree.new.fit(rows, residual, depth: depth)
          end
        end
        self
      end

      def predict(row)
        return base + @lr * trees.sum { |t| t.predict(row) } if @kind == :boosting
        trees.sum { |t| t.predict(row) }.to_f / trees.size
      end
    end
  end
end
