module EasyAILearning
  module Foundations
    # Small CART regression tree; labels 0/1 also give a classification score.
    class Tree
      attr_reader :root

      def fit(rows, targets, depth: 3, max_features: nil, seed: 1337)
        raise ArgumentError, "Invalid tree data" unless rows.size == targets.size && !rows.empty? && depth >= 0
        @features, @rng = max_features, Random.new(seed)
        @root = build(rows, targets, depth)
        self
      end

      def predict(row, node = root)
        return node.fetch(:value) if node.key?(:value)
        predict(row, row[node[:feature]] <= node[:threshold] ? node[:left] : node[:right])
      end

      private

      def build(rows, targets, depth)
        mean = targets.sum.to_f / targets.size
        return { value: mean } if depth.zero? || targets.uniq.size == 1
        features = rows.first.each_index.to_a
        features = features.sample(@features, random: @rng) if @features
        candidates = features.flat_map do |feature|
          values = rows.map { |r| r[feature] }.uniq.sort
          values.each_cons(2).map do |a, b|
            threshold = (a + b) / 2.0
            left, right = rows.each_index.partition { |i| rows[i][feature] <= threshold }
            [error(left.map { |i| targets[i] }) + error(right.map { |i| targets[i] }), feature, threshold, left, right]
          end
        end
        return { value: mean } if candidates.empty?
        _, feature, threshold, left, right = candidates.min_by(&:first)
        { feature: feature, threshold: threshold,
          left: build(left.map { |i| rows[i] }, left.map { |i| targets[i] }, depth - 1),
          right: build(right.map { |i| rows[i] }, right.map { |i| targets[i] }, depth - 1) }
      end

      def error(values)
        mean = values.sum.to_f / values.size
        values.sum { |v| (v - mean)**2 }
      end
    end
  end
end
