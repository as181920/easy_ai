module EasyAILearning
  module Foundations
    class Pca
      attr_reader :mean, :components, :eigenvalues

      def fit(rows, dimensions: 1, iterations: 100)
        raise ArgumentError, "Invalid PCA dimensions" unless !rows.empty? && dimensions.between?(1, rows.first.size)
        @mean = rows.transpose.map { |c| c.sum.to_f / c.size }
        centered = rows.map { |row| row.zip(mean).map { |a, b| a - b } }
        width = mean.size
        covariance = Array.new(width) { |i| Array.new(width) { |j| centered.sum { |r| r[i] * r[j] } / rows.size } }
        @components, @eigenvalues = [], []
        dimensions.times do |component|
          vector = Array.new(width) { |i| 1.0 / (i + component + 1) }
          iterations.times do
            candidate = covariance.map { |row| Math.dot(row, vector) }
            components.each do |basis|
              projection = Math.dot(candidate, basis)
              candidate = candidate.zip(basis).map { |a, b| a - projection * b }
            end
            norm = ::Math.sqrt(Math.dot(candidate, candidate))
            break if norm < 1e-12
            vector = candidate.map { |v| v / norm }
          end
          components.each do |basis|
            projection = Math.dot(vector, basis)
            vector = vector.zip(basis).map { |a, b| a - projection * b }
          end
          norm = ::Math.sqrt(Math.dot(vector, vector))
          if norm < 1e-10
            width.times do |axis|
              candidate = Array.new(width) { |i| i == axis ? 1.0 : 0.0 }
              components.each do |basis|
                projection = Math.dot(candidate, basis)
                candidate = candidate.zip(basis).map { |a, b| a - projection * b }
              end
              candidate_norm = ::Math.sqrt(Math.dot(candidate, candidate))
              if candidate_norm > 1e-10
                vector, norm = candidate, candidate_norm
                break
              end
            end
          end
          vector = vector.map { |v| v / norm }
          eigenvalue = Math.dot(vector, covariance.map { |row| Math.dot(row, vector) })
          components << vector
          eigenvalues << eigenvalue
        end
        self
      end

      def transform(rows)
        rows.map { |row| components.map { |c| Math.dot(row.zip(mean).map { |a, b| a - b }, c) } }
      end

      def reconstruct(encoded)
        encoded.map { |z| mean.each_index.map { |i| mean[i] + z.each_index.sum { |j| z[j] * components[j][i] } } }
      end
    end
  end
end
