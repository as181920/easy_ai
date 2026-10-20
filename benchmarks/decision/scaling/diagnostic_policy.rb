module DecisionScaling
  module_function

  # Development retention is a curriculum decision, not evidence of generalization.
  # Keep the frozen 90% experiments intact; new development decisions use 85%.
  def diagnostic_mastery_rules(original)
    original.merge("mixed_groups" => 0.85)
  end

  def prepare_development(root)
    prepare(root)
    path = File.join(root, "protocol.json")
    protocol = JSON.parse(File.read(path))
    protocol["mastery"] = diagnostic_mastery_rules(protocol.fetch("mastery"))
    protocol["development_policy"] = "User-authorized 85% retention floor, 2026-10-19; applied before any learning update. Historical copied protocols retain their own rules."
    write(path, protocol)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts "Development retention floor: 85%; two consecutive checks required"
  end

  def diagnostic_gate_history(history, rules)
    consecutive = 0
    first_stable = nil
    checks = history.map do |cell|
      canonical = cell["canonical"] || cell.fetch("profiles").fetch("binary").fetch("controlled").fetch("canonical")
      passed = DecisionRelational.known_mastery?(cell.fetch("training_probe"), canonical, rules)
      consecutive = passed ? consecutive + 1 : 0
      first_stable ||= cell.fetch("step") if consecutive >= rules.fetch("stable_checks")
      { "step" => cell.fetch("step"), "passed" => passed, "consecutive" => consecutive }
    end
    { "first_stable_step" => first_stable, "checks" => checks }
  end

  def diagnostic_policy_audit(root)
    protocol = verify(root)
    rules = protocol.fetch("mastery")
    paths = Dir[File.join(root, "{arms,supervised}", "*", "development.json")].sort
    histories = paths.to_h do |path|
      history = JSON.parse(File.read(path))
      [path.delete_prefix(root + "/"), { "history_sha256" => sha(path),
        "original_90" => diagnostic_gate_history(history, rules),
        "development_85" => diagnostic_gate_history(history, diagnostic_mastery_rules(rules)) }]
    end
    value = { "amendment" => "User-authorized development retention floor, 2026-10-19",
      "original_rules" => rules, "development_rules" => diagnostic_mastery_rules(rules), "histories" => histories,
      "scope" => "Retrospective gate assessment only; no optimizer updates or counterfactual model outcomes. Original protocols, trajectories, selections and release criteria remain historical evidence. Direction is assessed from held-out trends separately from release eligibility." }
    write(File.join(root, "diagnostic-policy-audit.json"), value)
    value
  end
end
