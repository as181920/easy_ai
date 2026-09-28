module EasyAI
  module Decision
    class Evaluator
      def initialize(predictor)
        @predictor = predictor
      end

      def collect(dataset)
        logits, targets, languages, groups, sources = [], [], [], [], []
        dataset.each.with_index do |example, index|
          logits << @predictor.logits(example)
          targets << example.target_index
          languages << example.language
          groups << example.group_id
          sources << example.source
          GC.start if ((index + 1) % 16).zero?
        end
        { logits: logits, targets: targets, languages: languages, groups: groups, sources: sources }
      end

      def evaluate(dataset)
        rows = collect(dataset)
        result = self.class.metrics(rows[:logits], rows[:targets], @predictor.calibrator)
        result["by_language"] = rows[:languages].uniq.to_h do |language|
          indexes = rows[:languages].each_index.select { |i| rows[:languages][i] == language }
          [language, self.class.metrics(indexes.map { |i| rows[:logits][i] }, indexes.map { |i| rows[:targets][i] }, @predictor.calibrator)]
        end
        result["by_source"] = rows[:sources].uniq.to_h do |source|
          indexes = rows[:sources].each_index.select { |i| rows[:sources][i] == source }
          [source, self.class.metrics(indexes.map { |i| rows[:logits][i] }, indexes.map { |i| rows[:targets][i] }, @predictor.calibrator)]
        end
        result["macro_source_accuracy"] = result["by_source"].values.sum { |metrics| metrics["accuracy"] } / result["by_source"].size
        result
      end

      def self.metrics(logits, targets, calibrator = Calibrator.new)
        raise ArgumentError, "Empty evaluation set" if logits.empty?
        probabilities = logits.map { |row| calibrator.probabilities(row) }
        correct = probabilities.each_with_index.map { |p, i| p.each_index.max_by { |j| p[j] } == targets[i] ? 1.0 : 0.0 }
        brier = probabilities.each_with_index.sum { |p, i| p.each_with_index.sum { |v, j| (v - (j == targets[i] ? 1 : 0))**2 } } / logits.length
        bins = Array.new(10) { [] }
        probabilities.each_with_index { |p, i| bins[[(p.max * 10).floor, 9].min] << [p.max, correct[i]] }
        reliability = bins.each_with_index.filter_map do |items, index|
          next if items.empty?
          { "lower" => index / 10.0, "count" => items.length, "confidence" => items.sum(&:first) / items.length,
           "accuracy" => items.sum(&:last) / items.length }
        end
        ece = reliability.sum { |bin| (bin["confidence"] - bin["accuracy"]).abs * bin["count"] / logits.length }
        { "count" => logits.length, "accuracy" => correct.sum / logits.length, "nll" => calibrator.nll(logits, targets),
         "brier" => brier, "ece" => ece, "reliability" => reliability }
      end
    end
  end
end
