module EasyAI
  module Decision
    class SemanticDiagnostics
      def initialize(predictor:, validation:, reference:, limit: 50, seed: 1337)
        raise ArgumentError, "limit must be positive" unless limit.is_a?(Integer) && limit > 0
        Data::Dataset.assert_disjoint!(validation, reference)
        @predictor, @reference, @seed = predictor, reference, seed
        @strata = validation.group_by { |example| [example.source, example.language] }.transform_values do |rows|
          groups = rows.map(&:group_id).uniq.first(limit).to_set
          rows.select { |row| groups.include?(row.group_id) }
        end
      end

      def evaluate
        counts = @reference.map { |row| [row.source, row.language, row.target] }.tally
        results = @strata.to_h do |(source, language), examples|
          rng = Random.new(@seed)
          states = examples.map(&:state).shuffle(random: rng)
          questions = examples.map(&:question).shuffle(random: rng)
          logits = %w[original shuffled_state shuffled_question].to_h do |condition|
            values = examples.each_with_index.map do |example, index|
              changed = example.to_h
              changed["state"] = states[index] if condition == "shuffled_state"
              changed["question"] = questions[index] if condition == "shuffled_question"
              result = @predictor.logits(Data::Example.new(changed))
              GC.start if ((index + 1) % 16).zero?
              result
            end
            [condition, values]
          end
          logits["label_frequency"] = examples.map do |example|
            example.options.map { |option| Math.log(counts.fetch([source, language, option["id"]], 0) + 1) }
          end
          original = logits.fetch("original").map { |row| row.each_index.max_by { |index| row[index] } }
          metrics = logits.to_h do |condition, rows|
            agreement = rows.each_with_index.count { |row, index| row.each_index.max_by { |i| row[i] } == original[index] }.fdiv(rows.size)
            [condition, Evaluator.metrics(rows, examples.map(&:target_index)).slice("count", "accuracy", "nll", "brier", "ece")
              .merge("prediction_agreement" => agreement)]
          end
          ["#{source}/#{language}", { "groups" => examples.map(&:group_id).uniq.size, "conditions" => metrics }]
        end
        { "scope" => "Validation only; uncalibrated logits. Shuffle within source/language, keep candidates and targets fixed. Sensitivity alone does not prove correctness.",
          "seed" => @seed, "by_source_language" => results }
      end
    end
  end
end
