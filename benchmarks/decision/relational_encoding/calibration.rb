module DecisionRelational
  module_function

  def confirmed_publication(root, variant)
    protocol = pilot_protocol(root, variant: variant)
    directory = pilot_directory(root, variant)
    selection = JSON.parse(File.read(File.join(directory, "selection.json")))
    profile = selection.fetch("profile")
    raise "No development-eligible profile" unless profile
    primary_path = File.join(directory, "seed-1337/summary.json")
    raise "Selected pilot changed" unless sha(primary_path) == selection.fetch("summary_sha256")
    summaries = [primary_path, File.join(directory, "seed-2027/summary.json")].map { |path| JSON.parse(File.read(path)) }
    raise "Recipe did not confirm on the same profile" unless summaries.all? { |summary| summary.fetch("known_mastery_passed") && summary.fetch("selected").key?(profile) }
    baseline = pilot_baseline(root, variant: variant)
    unless summaries.all? { |summary| pilot_eligible?(summary.fetch("selected").fetch(profile).fetch("metrics"), baseline.fetch(profile), profile, protocol.fetch("publication")) }
      raise "Selected development metrics fail publication guards"
    end
    { "profile" => profile, "checkpoints" => { "parent" => PARENT,
      "candidate" => summaries.first.fetch("selected").fetch(profile).fetch("checkpoint"),
      "confirmation" => summaries.last.fetch("selected").fetch(profile).fetch("checkpoint") },
      "summary_sha256" => [primary_path, File.join(directory, "seed-2027/summary.json")].to_h { |path| [File.basename(File.dirname(path)), sha(path)] } }
  end

  def calibrate(root, variant: "standard")
    protocol = pilot_protocol(root, variant: variant)
    directory = pilot_directory(root, variant)
    raise "Calibration already frozen" if File.exist?(File.join(directory, "calibration.json"))
    publication = confirmed_publication(root, variant)
    profile = publication.fetch("profile")
    raw = DecisionJudgment.panel(directory, "calibration", profile)
    policies = publication.fetch("checkpoints").to_h do |name, path|
      loaded = EasyAI::Decision::Checkpoint.load(path)
      examples, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), raw, DecisionRobust.device)
      calibrator = EasyAI::Decision::Calibrator.new.fit(logits, examples.map(&:target_index))
      value = { "checkpoint" => loaded.fetch(:path), "weights_sha256" => loaded.fetch(:weights_fingerprint), "temperature" => calibrator.temperature,
        "calibration_raw" => DecisionJudgment.metrics(raw, logits), "calibration_adjusted" => DecisionJudgment.metrics(raw, logits, temperature: calibrator.temperature) }
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      [name, value]
    end
    value = publication.except("checkpoints").merge("policies" => policies,
      "protocol_sha256" => sha(File.join(directory, "protocol.json")), "selection_sha256" => sha(File.join(directory, "selection.json")),
      "data_sha256" => protocol.fetch("files_sha256"),
      "capability" => "Shared EN/ZH short-record factual scoring with canonical candidate descriptions. Binary is conditional on known evidence and cannot detect answerability; three-state is limited to the measured explicit-record contract. Other languages, longer records and arbitrary wording are unvalidated.")
    write(File.join(directory, "calibration.json"), value)
    write(File.join(directory, "calibration-sha256.json"), { "sha256" => sha(File.join(directory, "calibration.json")) })
    value
  end

  def frozen_acceptance(root, variant)
    protocol = pilot_protocol(root, variant: variant)
    publication = confirmed_publication(root, variant)
    directory = pilot_directory(root, variant)
    path = File.join(directory, "calibration.json")
    value = JSON.parse(File.read(path))
    expected = JSON.parse(File.read(File.join(directory, "calibration-sha256.json"))).fetch("sha256")
    raise "Frozen calibration changed" unless sha(path) == expected
    raise "Calibrated profile changed" unless value.fetch("profile") == publication.fetch("profile")
    raise "Calibrated data manifest changed" unless value.fetch("data_sha256") == protocol.fetch("files_sha256")
    raise "Calibrated case set changed" unless value.fetch("policies").keys.sort == publication.fetch("checkpoints").keys.sort
    publication.fetch("checkpoints").each do |name, checkpoint|
      raise "Calibrated checkpoint changed" unless value.fetch("policies").fetch(name).fetch("checkpoint") == EasyAI::Decision::Checkpoint.resolve(checkpoint)
    end
    raise "Protocol changed after calibration" unless value.fetch("protocol_sha256") == sha(File.join(directory, "protocol.json"))
    raise "Selection changed after calibration" unless value.fetch("selection_sha256") == sha(File.join(directory, "selection.json"))
    value.fetch("summary_sha256").each do |seed, digest|
      raise "Confirmation summary changed" unless sha(File.join(directory, seed, "summary.json")) == digest
    end
    marker_path = File.join(directory, "acceptance-opened.json")
    if File.exist?(marker_path)
      marker = JSON.parse(File.read(marker_path))
      raise "Frozen calibration policy changed" unless marker.fetch("calibration_sha256") == sha(path)
    else
      write(marker_path, { "at" => Time.now.utc.iso8601, "calibration_sha256" => sha(path), "protocol_sha256" => value.fetch("protocol_sha256"),
        "scope" => "Open reserved acceptance once for fixed checkpoints/profile/policies; interruption recovery may evaluate only these same frozen choices." })
    end
    value
  end
end
