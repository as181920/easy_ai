require "json"

module EasyAI
  module Decision
    class RelationEvaluation
      def initialize(predictor, batch_size: 16)
        raise ArgumentError, "batch_size must be positive" unless batch_size.is_a?(Integer) && batch_size > 0
        @predictor, @batch_size = predictor, batch_size
        @collator = Data::Collator.new(tokenizer: predictor.tokenizer, config: predictor.model.config)
        @policy = Runtime::DevicePolicy.new(budget_mib: predictor.model.config[:runtime]["gpu_memory_budget_mib"])
      end

      def evaluate(dataset, controls: false)
        examples = dataset.to_a
        metadata = File.foreach(dataset.path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line).fetch("relation") }
        unless metadata.all? { |row| row["binding_group"] && row.fetch("checks").key?("subject_switch") }
          raise ArgumentError, "Relation evaluation requires v3 metadata; prepare a new relations-v3 dataset"
        end
        logits = collect(examples)
        result = metrics(examples, metadata, logits)
        reversed = examples.map { |example| Data::Example.new(example.to_h.merge("options" => example.options.reverse)) }
        permuted = collect(reversed)
        permutation_error = logits.each_index.map { |i| logits[i].zip(permuted[i].reverse).map { |a, b| (a - b).abs }.max }.max
        result["maximum_permutation_logit_error"] = permutation_error
        result["dataset_sha256"] = dataset.fingerprint
        result["calibrated"] = @predictor.calibrated
        result["requested_batch_size"] = @batch_size
        result["failures"] = examples.each_index.filter_map do |i|
          chosen = examples[i].options[logits[i].each_index.max_by { |j| logits[i][j] }].fetch("id")
          examples[i].to_h.merge("chosen" => chosen, "logits" => logits[i]) if chosen != examples[i].target
        end.first(20)
        result["by_language"] = examples.map(&:language).uniq.to_h do |language|
          indices = examples.each_index.select { |i| examples[i].language == language }
          [language, metrics(indices.map { |i| examples[i] }, indices.map { |i| metadata[i] }, indices.map { |i| logits[i] })]
        end
        result["by_fact_pattern"] = metadata.each_index.group_by do |i|
          metadata[i].fetch("facts").uniq.length == 1 ? "same_truth" : "mixed_truth"
        end.transform_values do |indices|
          Evaluator.metrics(indices.map { |i| logits[i] }, indices.map { |i| examples[i].target_index }, @predictor.calibrator)
            .slice("count", "accuracy", "nll", "brier", "ece")
        end
        if controls
          result["controls"] = %w[state question].to_h do |field|
            changed = examples.map { |example| Data::Example.new(example.to_h.merge(field => "信息未提供。 No information.")) }
            rows = collect(changed)
            ["constant_#{field}", Evaluator.metrics(rows, examples.map(&:target_index), @predictor.calibrator).slice("accuracy", "nll")]
          end
        end
        result["device"] = @predictor.device
        result
      end

      private

      def collect(examples)
        @predictor.model.eval
        examples.each_slice(@batch_size).flat_map do |rows|
          collect_batch(rows)
        end
      end

      def collect_batch(rows)
        result = batch_logits(rows)
        GC.start
        @policy.check_budget!(@predictor.device)
        result
      rescue Torch::Error, Runtime::DevicePolicy::MemoryBudgetExceeded => error
        raise unless @predictor.device == "cuda" && @policy.recoverable?(error)
        GC.start
        return [@predictor.logits(rows.first)] if rows.length == 1
        rows.each_slice([rows.length / 2, 1].max).flat_map { |part| collect_batch(part) }
      end

      def batch_logits(rows)
        Torch.no_grad do
          batch = @collator.call(rows, device: @predictor.device)
          @predictor.model.call(batch).cpu.to_a
        end
      end

      def metrics(examples, metadata, logits)
        result = Evaluator.metrics(logits, examples.map(&:target_index), @predictor.calibrator)
        chosen = logits.each_with_index.map { |row, i| examples[i].options[row.each_index.max_by { |j| row[j] }].fetch("id") }
        correct = examples.each_index.map { |i| chosen[i] == examples[i].target }
        result["pairs"] = %w[question_flip fact_flip irrelevant_fact order subject_switch role_swap].to_h do |kind|
          indices = metadata.each_index.select { |i| kind != "role_swap" || metadata[i].fetch("facts").uniq.length == 2 }
          groups = indices.group_by { |i| metadata[i].fetch("checks").fetch(kind) }.values
          [kind, pair_metrics(kind, groups, examples, metadata, chosen, correct)]
        end
        groups = metadata.each_index.group_by { |i| metadata[i].fetch("binding_group") }.values
        raise ArgumentError, "Incomplete binding groups" unless groups.all? { |indices| indices.size == 4 }
        result["groups"] = { "binding" => group_metrics(groups, correct) }
        %w[same_truth mixed_truth].each do |pattern|
          subset = groups.select { |indices| (metadata[indices.first].fetch("facts").uniq.length == 2) == (pattern == "mixed_truth") }
          result["groups"]["binding"][pattern] = group_metrics(subset, correct)
        end
        result.slice("count", "accuracy", "nll", "brier", "ece", "pairs", "groups")
      end

      def pair_metrics(kind, groups, examples, metadata, chosen, correct)
        raise ArgumentError, "Incomplete relation pairs: #{kind}" unless groups.all? { |indices| indices.size == 2 }
        flips = groups.map do |a, _b|
          kind == "subject_switch" ? metadata[a].fetch("facts").uniq.length == 2 : %w[question_flip fact_flip role_swap].include?(kind)
        end
        unless groups.zip(flips).all? { |(a, b), flip| (examples[a].target != examples[b].target) == flip }
          raise ArgumentError, "Inconsistent relation labels: #{kind}"
        end
        result = { "count" => groups.size,
          "both_correct" => fraction(groups) { |a, b| correct[a] && correct[b] },
          "expected_relation_rate" => fraction(groups.zip(flips)) { |(a, b), flip| (chosen[a] != chosen[b]) == flip } }
        if kind == "subject_switch"
          %w[same_truth mixed_truth].each do |pattern|
            subset = groups.zip(flips).select { |_pair, flip| flip == (pattern == "mixed_truth") }
            result[pattern] = { "count" => subset.size,
              "both_correct" => fraction(subset) { |(a, b), _flip| correct[a] && correct[b] },
              "expected_relation_rate" => fraction(subset) { |(a, b), flip| (chosen[a] != chosen[b]) == flip } }
          end
        end
        result
      end

      def group_metrics(groups, correct)
        { "count" => groups.size, "all_correct" => fraction(groups) { |indices| indices.all? { |i| correct[i] } } }
      end

      def fraction(items, &block)
        items.empty? ? nil : items.count(&block).fdiv(items.size)
      end
    end
  end
end
