module DecisionScaling
  module_function

  def select_supervised(root)
    protocol = supervised_protocol(root)
    path = File.join(root, "supervised-selection.json")
    raise "Supervised selection already frozen" if File.exist?(path)
    summaries = protocol.fetch("profiles").to_h do |profile|
      file = File.join(root, "supervised", "#{profile}-#{protocol.fetch('scale')}-1337", "summary.json")
      value = JSON.parse(File.read(file))
      terminal = value.fetch("step") == protocol.fetch("steps") || (value.fetch("step") == SupervisedTrainer::SIGNATURE.fetch("known_updates") && !value.fetch("known_passed"))
      raise "Supervised primary recipe incomplete" unless terminal
      [profile, { "value" => value, "sha256" => sha(file) }]
    end
    choices = summaries.flat_map do |capacity, item|
      next [] unless item.fetch("value").fetch("known_passed")
      item.fetch("value").fetch("selected").map { |profile, cell| [capacity, profile, cell] }
    end
    choice = choices.max_by { |capacity, profile, cell| [profile == "joint" ? 1 : 0, *cell.fetch("key"), capacity == "base" ? 1 : 0] }
    capacity, profile, cell = choice
    value = { "eligible" => !choice.nil?, "cell" => choice ? "#{capacity}-#{protocol.fetch('scale')}" : nil, "profile" => profile,
      "controlled_steps" => choice ? protocol.fetch("initial_checkpoints").fetch(capacity).fetch("steps") : nil,
      "selected" => cell, "summary_sha256" => summaries.transform_values { |item| item.fetch("sha256") },
      "protocol_sha256" => sha(File.join(root, "supervised-protocol.json")),
      "scope" => "All permitted primary recipes are terminal. Eligible joint profile is preferred, otherwise eligible binary. A negative selection skips confirmation/calibration/acceptance; a positive selection still requires the same full recipe at seed 2027." }
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end

  def supervised_selection(root)
    protocol = supervised_protocol(root)
    path = File.join(root, "supervised-selection.json")
    raise "Supervised selection changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    value = JSON.parse(File.read(path))
    raise "Supervised selection protocol changed" unless value.fetch("protocol_sha256") == sha(File.join(root, "supervised-protocol.json"))
    value.fetch("summary_sha256").each do |profile, digest|
      file = File.join(root, "supervised", "#{profile}-#{protocol.fetch('scale')}-1337/summary.json")
      raise "Supervised primary summary changed" unless sha(file) == digest
    end
    value
  end
end
