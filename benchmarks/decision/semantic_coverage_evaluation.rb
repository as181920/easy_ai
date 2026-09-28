require_relative "../../lib/easy_ai"

# Batched offline evaluation keeps every temporary tensor inside one method call.
# This evaluates the existing neural model; no text rules compute an answer.
class SemanticCoverageEvaluation
  def initialize(model:, tokenizer:, device: "cpu", batch_size: 16)
    @model, @device, @batch_size = model, device, batch_size
    @model.to(device).eval
    @collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: model.config)
  end

  def collect(rows)
    rows.each_slice(@batch_size).flat_map do |batch|
      values = batch_logits(batch)
      GC.start
      values
    end
  end

  def evaluate(rows, permutations: false)
    logits = collect(rows)
    result = metrics(rows, logits)
    if permutations
      reversed = rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) }
      other = collect(reversed)
      pairs = rows.each_index.map do |index|
        predicted = rows[index].options[logits[index].each_index.max_by { |i| logits[index][i] }].fetch("id")
        reverse_predicted = reversed[index].options[other[index].each_index.max_by { |i| other[index][i] }].fetch("id")
        { "id" => rows[index].id, "source" => rows[index].source, "language" => rows[index].language,
          "target" => rows[index].target, "prediction" => predicted, "reversed_prediction" => reverse_predicted }
      end
      result["reversed"] = metrics(reversed, other)
      result["permutation_agreement"] = pairs.count { |row| row["prediction"] == row["reversed_prediction"] }.fdiv(pairs.size)
      result["both_correct"] = pairs.count { |row| row["prediction"] == row["target"] && row["reversed_prediction"] == row["target"] }.fdiv(pairs.size)
      result["predictions"] = pairs
    end
    result
  end

  def diagnostics(rows)
    conditions = { "original" => rows }
    %w[state question].each do |field|
      changed = rows.group_by { |row| [row.source, row.language] }.values.flat_map do |group|
        values = group.map { |row| row.public_send(field) }.shuffle(random: Random.new(1337))
        group.each_with_index.map { |row, index| EasyAI::Decision::Data::Example.new(row.to_h.merge(field => values[index])) }
      end
      conditions["shuffled_#{field}"] = changed
    end
    { "scope" => "Validation sensitivity diagnostic. Perturbed inputs retain original labels; these are not new gold examples. Sensitivity alone is not correctness.",
      "conditions" => conditions.transform_values { |examples| evaluate(examples) } }
  end

  private

  def batch_logits(rows)
    Torch.no_grad do
      values = @model.call(@collator.call(rows, device: @device)).to_a
      rows.each_with_index.map { |row, index| values[index].first(row.options.size) }
    end
  end

  def metrics(rows, logits)
    result = EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index))
    { "by_source" => :source, "by_language" => :language }.each do |name, field|
      result[name] = rows.each_index.group_by { |index| rows[index].public_send(field) }.transform_values do |indexes|
        EasyAI::Decision::Evaluator.metrics(indexes.map { |i| logits[i] }, indexes.map { |i| rows[i].target_index })
      end
    end
    result["macro_source_accuracy"] = result["by_source"].values.sum { |row| row["accuracy"] }.fdiv(result["by_source"].size)
    result
  end
end
