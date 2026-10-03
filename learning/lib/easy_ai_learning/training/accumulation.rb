module EasyAILearning
  module Training
    module Accumulation
      module_function

      # Each objective returns a mean loss; weights are actual sample/token counts.
      def backward(objectives, counts:)
        raise ArgumentError, "Invalid microbatch counts" unless objectives.size == counts.size && counts.all? { |v| v > 0 }
        total, value = counts.sum.to_f, 0.0
        objectives.zip(counts).each do |objective, count|
          loss = objective.call * (count / total)
          value += loss.item
          loss.backward
        end
        value
      end
    end
  end
end
