module EasyAI
  module Decision
    # A signed logit margin for recorded complementary factual labels, not text rules.
    module PairMarginLoss
      module_function

      def call(logits, examples, margin: 1.0)
        terms = examples.each_with_index.filter_map do |row, index|
          next unless row.contrast_groups["fact_flip"]
          ids = row.options.map { |option| option.fetch("id") }
          unless %w[yes no].include?(row.target) && (%w[yes no] - ids).empty?
            raise ArgumentError, "Fact-flip margin requires complementary yes/no labels"
          end
          signed = logits[index][ids.index("yes")] - logits[index][ids.index("no")]
          signed = -signed if row.target == "no"
          Torch::NN::Functional.relu(margin - signed)
        end
        return logits.sum * 0 if terms.empty?
        # Zero contribution for unpaired rows; match CE's per-example normalization.
        Torch.stack(terms).sum / examples.size
      end
    end
  end
end
