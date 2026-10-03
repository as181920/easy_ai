module EasyAILearning
  module Foundations
    class KMeans
      attr_reader :centers

      def fit(rows, clusters: 2, steps: 20, seed: 1337)
        raise ArgumentError, "Invalid cluster count" unless clusters.between?(1, rows.size)
        @centers = rows.sample(clusters, random: Random.new(seed)).map(&:dup)
        steps.times do
          assigned = rows.group_by { |row| predict(row) }
          @centers = centers.each_index.map do |i|
            group = assigned[i]
            group ? group.transpose.map { |c| c.sum.to_f / c.size } : centers[i]
          end
        end
        self
      end

      def predict(row)
        centers.each_index.min_by { |i| row.zip(centers[i]).sum { |a, b| (a - b)**2 } }
      end
    end
  end
end
