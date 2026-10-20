module DecisionScaling
  module_function

  def minimum_groups(metrics)
    metrics.fetch("by_language").values.flat_map { |cell| %w[actor event].map { |axis| cell.fetch("#{axis}_binding_all_correct") } }.min
  end

  def extension_needed?(screen, history)
    return true unless screen.fetch("known_mastery_passed")
    previous = history.find { |cell| cell.fetch("step") == 1000 }
    raise "Incomplete extension trend history" unless previous && history.last.fetch("step") == 1500
    latest = history.last.fetch("canonical")
    latest.fetch("worst_language_balanced") + 1e-12 >= previous.dig("canonical", "worst_language_balanced") + 0.02 ||
      minimum_groups(latest) + 1e-12 >= minimum_groups(previous.fetch("canonical")) + 0.05
  end

  def select_extensions(root)
    protocol = verify(root)
    path = File.join(root, "extension-selection.json")
    raise "Extensions already frozen" if File.exist?(path)
    inputs = protocol.fetch("profiles").product([32, 512]).to_h do |profile, scale|
      file = File.join(root, "arms", "#{profile}-#{scale}-1337/screen.json")
      value = JSON.parse(File.read(file))
      raise "Initial factorial incomplete" unless value.fetch("step") == 1500
      ["#{profile}-#{scale}", { "path" => file, "result" => value }]
    end
    reasons = protocol.fetch("profiles").to_h do |profile|
      item = inputs.fetch("#{profile}-512")
      history = JSON.parse(File.read(File.join(File.dirname(item.fetch("path")), "development.json")))
      [profile, extension_needed?(item.fetch("result"), history)]
    end
    effects = protocol.fetch("profiles").map { |profile| inputs.fetch("#{profile}-512").dig("result", "development", "canonical", "worst_language_balanced") - inputs.fetch("#{profile}-32").dig("result", "development", "canonical", "worst_language_balanced") }
    midpoint = effects.size == 2 && effects.all? { |value| value.abs >= 0.03 } && effects.reduce(:*) < 0
    value = { "extend" => reasons.values.any? ? protocol.fetch("profiles").map { |profile| "#{profile}-512" } : [],
      "reasons" => reasons, "midpoint" => midpoint, "data_effect_by_profile" => protocol.fetch("profiles").zip(effects).to_h,
      "screen_sha256" => inputs.transform_values { |item| sha(item.fetch("path")) },
      "scope" => "Both 512-world capacities extend together if either is still learning by the preregistered criterion; initial factorial remains a fixed-visit comparison." }
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end

  def final_controlled_selection(root)
    protocol = verify(root)
    path = File.join(root, "selection.json")
    raise "Controlled selection already frozen" if File.exist?(path)
    decision_path = File.join(root, "extension-selection.json")
    decision = extension_decision(root)
    cells = protocol.fetch("profiles").product([32, 512]).map { |profile, scale| "#{profile}-#{scale}" }
    cells << "base-128" if decision.fetch("midpoint")
    results = cells.to_h do |cell|
      suffix = decision.fetch("extend").include?(cell) ? "extended.json" : "screen.json"
      file = File.join(root, "arms", "#{cell}-1337", suffix)
      [cell, JSON.parse(File.read(file)).merge("result_sha256" => sha(file))]
    end
    qualified = results.select { |_, value| value.fetch("known_mastery_passed") }
    winner = qualified.max_by do |_, value|
      required = protocol.fetch("required_development").map { |name| value.fetch("development").fetch(name) }
      [value.fetch("controlled_eligible") ? 1 : 0, required.map { |metrics| minimum_groups(metrics) }.min,
        required.map { |metrics| metrics.fetch("worst_language_balanced") }.min, -required.sum { |metrics| metrics.fetch("nll") }, -value.fetch("parameters")]
    end
    value = { "cell" => winner&.first, "controlled_steps" => winner&.last&.fetch("step"),
      "checkpoint" => winner&.last&.fetch("checkpoint"), "controlled_eligible" => winner ? winner.last.fetch("controlled_eligible") : false,
      "results_sha256" => results.transform_values { |item| item.fetch("result_sha256") }, "extension_sha256" => sha(decision_path),
      "scope" => "A known-mastery-qualified controlled curriculum may proceed to separately frozen natural/unknown supervision. This selection is not a factual release or full-recipe confirmation." }
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end

  def extension_decision(root)
    path = File.join(root, "extension-selection.json")
    raise "Extension decision changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    value = JSON.parse(File.read(path))
    value.fetch("screen_sha256").each do |cell, digest|
      raise "Initial scale screen changed" unless sha(File.join(root, "arms", "#{cell}-1337/screen.json")) == digest
    end
    value
  end
end
