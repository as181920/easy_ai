module EasyAILearning
  module Transformer
    module Sinusoidal
      module_function

      def values(length, width)
        Array.new(length) do |position|
          Array.new(width) do |column|
            angle = position / 10000.0**(2 * (column / 2).to_f / width)
            column.even? ? ::Math.sin(angle) : ::Math.cos(angle)
          end
        end
      end
    end
  end
end
