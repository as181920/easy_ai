module EasyAI
  module Decision
    # Couple opposite-label actor queries; CE still anchors absolute decisions.
    module BindingPairLoss
      module_function

      def call(logits, examples, margin: 1.0)
        groups = examples.each_index.select { |index| examples[index].source.end_with?("Actor") }.group_by do |index|
          examples[index].contrast_groups.fetch("binding")
        end
        raise ArgumentError, "Incomplete binding pair" unless groups.values.all? { |indices| indices.size.even? }
        terms = groups.values.flat_map { |indices| indices.each_slice(2).to_a }.filter_map do |indices|
          pair = indices.map { |index| examples[index] }
          unless pair.map(&:id).uniq.size == 2 && pair.map(&:state).uniq.size == 1
            raise ArgumentError, "Invalid binding pair membership"
          end
          next unless indices.map { |index| examples[index].target }.sort == %w[no yes]
          signed = indices.to_h do |index|
            row = examples[index]
            ids = row.options.map { |option| option.fetch("id") }
            raise ArgumentError, "Binding loss needs binary yes/no options" unless ids.sort == %w[no yes]
            [row.target, logits[index][ids.index("yes")] - logits[index][ids.index("no")]]
          end
          Torch::NN::Functional.softplus(margin - (signed.fetch("yes") - signed.fetch("no")))
        end
        terms.empty? ? logits.sum * 0 : Torch.stack(terms).mean
      end
    end
  end
end
