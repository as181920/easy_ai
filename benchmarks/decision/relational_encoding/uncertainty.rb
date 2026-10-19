module DecisionRelational
  module_function

  def paired_uncertainty(raw, parent, candidate, samples: 1000, seed: 1337)
    unless !raw.empty? && raw.size == parent.size && raw.size == candidate.size && samples.is_a?(Integer) && samples.positive?
      raise ArgumentError, "Invalid paired uncertainty inputs"
    end
    examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    cells = raw.each_index.group_by { |index| raw[index].fetch("language") }.transform_values do |indexes|
      frames = indexes.group_by { |index| raw[index].fetch("group_id") }.values.map do |items|
        counts = [parent, candidate].map { |logits| items.count { |index| logits[index].each_index.max_by { |j| logits[index][j] } == examples[index].target_index } }
        [*counts, items.size]
      end
      rng = Random.new(seed)
      draws = Array.new(samples) do
        selected = Array.new(frames.size) { frames.sample(random: rng) }
        (selected.sum { |a, b, _| b - a }).fdiv(selected.sum(&:last))
      end.sort
      { "families" => frames.size, "rows" => indexes.size, "paired_accuracy_gain_interval95" => [draws[((samples - 1) * 0.025).floor], draws[((samples - 1) * 0.975).floor]] }
    end
    { "by_language" => cells, "samples" => samples, "seed" => seed,
      "scope" => "Paired family-cluster bootstrap for row-weighted accuracy gain; related views do not add independent families. Small family counts remain weak evidence." }
  end
end
