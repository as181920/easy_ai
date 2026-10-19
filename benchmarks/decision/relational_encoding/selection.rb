module DecisionRelational
  module_function

  def fit_selection_path(root, selection)
    raise ArgumentError, "Unknown fitting selection" unless %w[default candidate].include?(selection.to_s)
    File.join(root, selection.to_s == "default" ? "fit-selection.json" : "fit-selection-candidate.json")
  end

  def select_fit(root, selection: "default")
    verify(root)
    path = fit_selection_path(root, selection)
    raise "Fit selection already frozen" if File.exist?(path)
    names = ARMS + RECIPES.keys.select { |name| File.exist?(File.join(root, "recipes", "#{name}.json")) }
    names = %w[joint-rotary-candidate separate-rotary] if selection.to_s == "candidate"
    results = names.to_h do |name|
      result_path = File.join(root, "fit-#{name}-1337/result.json")
      raise "Registered fit incomplete: #{name}" unless File.exist?(result_path)
      [name, JSON.parse(File.read(result_path))]
    end
    qualified = results.select { |_, result| result.fetch("passed") }
    winner = qualified.max_by do |name, result|
      value = result.fetch("observed_development")
      groups = value.fetch("by_language").values.flat_map { |cell| %w[actor event].map { |kind| cell.fetch("#{kind}_binding_all_correct") } }
      [value.fetch("worst_language_balanced"), groups.min, -value.fetch("nll"), name == "separate" ? 1 : 0]
    end
    value = { "winner" => winner&.first, "checkpoint" => winner&.last&.fetch("checkpoint"),
      "results_sha256" => names.to_h { |name| [name, sha(File.join(root, "fit-#{name}-1337/result.json"))] },
      "scope" => "Selection among fitting-qualified recipes using observed development, before second-seed confirmation; no acceptance opened" }
    write(path, value)
    value
  end

  def check_confirmation!(root, arm, selection: "default")
    selection = JSON.parse(File.read(fit_selection_path(root, selection)))
    raise "Recipe not selected for confirmation" unless selection.fetch("winner") == arm
    selection.fetch("results_sha256").each do |name, digest|
      raise "Selected result changed" unless sha(File.join(root, "fit-#{name}-1337/result.json")) == digest
    end
  end
end
