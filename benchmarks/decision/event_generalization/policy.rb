module DecisionEventGeneralization
  module_function

  def direction?(candidate, reference)
    event_min = ->(value) { %w[events expressions].flat_map { |view| value.fetch("development").fetch(view).fetch("by_language").values.map { |cell| cell.fetch("event_binding_all_correct") } }.min }
    actor_min = lambda do |value|
      panels = value.fetch("development").slice("combinations", "actors", "events", "expressions").values + value.fetch("regression").values
      panels.flat_map { |panel| panel.fetch("by_language").values.map { |cell| cell.fetch("actor_binding_all_correct") } }.min
    end
    event_min.call(candidate) + 1e-12 >= event_min.call(reference) + 0.05 && actor_min.call(candidate) + 0.05 + 1e-12 >= actor_min.call(reference) &&
      %w[combinations actors events expressions].all? do |view|
        Catalog::LANGUAGES.all? { |language| candidate.dig("development", view, "by_language", language, "balanced") + 0.03 + 1e-12 >= reference.dig("development", view, "by_language", language, "balanced") }
      end
  end

  def rank(result)
    primary = result.fetch("development").slice("events", "expressions").values
    required = result.fetch("development").except("composed").values
    [primary.flat_map { |panel| panel.fetch("by_language").values.map { |cell| cell.fetch("event_binding_all_correct") } }.min,
      (required + result.fetch("regression").values).flat_map { |panel| panel.fetch("by_language").values.map { |cell| cell.fetch("actor_binding_all_correct") } }.min,
      required.map { |panel| panel.fetch("worst_language_balanced") }.min, -required.sum { |panel| panel.fetch("nll") }]
  end

  def independent_pairs(rows, predictions, axis:)
    raise "Pair prediction count mismatch" unless rows.size == predictions.size
    selected = rows.each_index.select do |index|
      facts = rows[index].fetch("world").fetch("facts")
      next false unless facts.size == 2 && facts.map { |fact| fact.fetch("truth") }.uniq.size == 2
      actual = facts.map { |fact| fact.fetch("actor") }.uniq.size == 1 ? "event" : "actor"
      actual == axis.to_s
    end
    selected.group_by { |index| rows[index].fetch("language") }.transform_values do |indices|
      pairs = indices.group_by { |index| [rows[index].fetch("state"), rows[index].dig("world", "assertion")] }.values
      raise "Independent pair membership invalid" unless pairs.all? { |pair| pair.size == 2 && pair.map { |index| rows[index].fetch("target") }.sort == %w[no yes] }
      counts = pairs.map { |pair| pair.count { |index| predictions[index] == rows[index].fetch("target") } }.tally
      { "groups" => pairs.size, "rows" => indices.size, "row_accuracy" => indices.count { |index| predictions[index] == rows[index].fetch("target") }.fdiv(indices.size),
        "zero_correct" => counts.fetch(0, 0), "one_correct" => counts.fetch(1, 0), "two_correct" => counts.fetch(2, 0),
        "all_correct" => counts.fetch(2, 0).fdiv(pairs.size), "same_prediction" => pairs.count { |pair| pair.map { |index| predictions[index] }.uniq.size == 1 } }
    end
  end

  def audit_rows!(rows)
    raise "Duplicate decision IDs" unless rows.map { |row| row.fetch("id") }.uniq.size == rows.size
    inputs = {}
    rows.each do |row|
      EasyAI::Decision::Data::Example.new(row)
      Corpus.gold!(row)
      world = row.fetch("world")
      raise "Invalid explicit truth" unless world.fetch("facts").all? { |fact| [true, false].include?(fact.fetch("truth")) } && [true, false].include?(world.fetch("assertion"))
      raise "Mixed truth metadata incorrect" unless world.fetch("mixed") == (world.fetch("facts").map { |fact| fact.fetch("truth") }.uniq.size == 2)
      rendered = Catalog.render(row, form: world.fetch("form"))
      raise "Renderer/gold material mismatch" unless row.values_at("state", "question") == rendered.values_at("state", "question")
      key = Digest::SHA256.hexdigest(JSON.generate(row.values_at("state", "question", "options")))
      raise "Conflicting identical inputs" if inputs.key?(key) && inputs.fetch(key) != row.fetch("target")
      inputs[key] = row.fetch("target")
    end
    %w[binding fact_flip question_flip].each do |axis|
      groups = rows.group_by { |row| row.fetch("contrast_groups").fetch(axis) }
      raise "Incomplete #{axis} counterfactual" unless groups.values.all? { |group| group.size == 2 }
    end
    true
  end
end
