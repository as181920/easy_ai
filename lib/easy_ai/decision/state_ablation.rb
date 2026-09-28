module EasyAI
  module Decision
    # Diagnostic on validation data, never a replacement for held-out task evaluation.
    class StateAblation
      def initialize(predictor:, validation:, reference:, language: nil, limit: 200, seed: 20260928)
        raise ArgumentError, "Diagnostic limit must be positive" unless limit.is_a?(Integer) && limit > 0
        Data::Dataset.assert_disjoint!(validation, reference)
        @predictor, @reference, @seed = predictor, reference, seed
        languages = validation.map(&:language)
        @language = language || (languages.include?("zh-CN") ? "zh-CN" : languages.first)
        @examples = validation.select { |example| example.language == @language }.uniq(&:group_id).first(limit)
        raise ArgumentError, "No validation examples for #{@language}" if @examples.empty?
      end

      def evaluate
        counts = @reference.select { |example| example.language == @language }.map { |example| [example.question, example.target] }.tally
        shuffled = @examples.map(&:state).shuffle(random: Random.new(@seed))
        rows = %w[original shuffled_state constant_state].to_h do |condition|
          logits = @examples.each_with_index.map do |example, index|
            state = case condition
                    when "original" then example.state
                    when "shuffled_state" then shuffled[index]
                    else "无上下文"
                    end
            @predictor.logits(Data::Example.new(example.to_h.merge("state" => state)))
          end
          [condition, logits]
        end
        rows["label_frequency"] = @examples.map do |example|
          example.options.map { |option| Math.log(counts.fetch([example.question, option.fetch("id")], 0) + 1) }
        end
        rows["uniform"] = @examples.map { |example| Array.new(example.options.length, 0.0) }
        calibrator = Calibrator.new
        original = rows.fetch("original").map { |row| calibrator.probabilities(row) }
        targets = @examples.map(&:target_index)
        conditions = rows.to_h do |condition, logits|
          probabilities = logits.map { |row| calibrator.probabilities(row) }
          differences = probabilities.zip(original).map { |left, right| left.zip(right).map { |a, b| (a - b).abs }.max }
          agreement = probabilities.zip(original).count { |left, right| argmax(left) == argmax(right) }.fdiv(@examples.size)
          metrics = Evaluator.metrics(logits, targets).slice("count", "accuracy", "nll", "brier", "ece")
          [condition, metrics.merge("prediction_agreement_with_original" => agreement,
            "mean_max_probability_change" => differences.sum / differences.size, "max_probability_change" => differences.max)]
        end
        { "scope" => "#{@language} validation; uncalibrated logits; fixed candidates across state interventions",
          "groups" => @examples.size, "train_groups" => @reference.groups.size, "shuffle_seed" => @seed,
          "uniform_note" => "Uniform accuracy uses first-option argmax; expected random-sampling accuracy is mean(1/K).",
          "expected_random_accuracy" => @examples.sum { |example| 1.0 / example.options.size } / @examples.size,
          "conditions" => conditions }
      end

      private

      def argmax(values)
        values.each_index.max_by { |index| values[index] }
      end
    end
  end
end
