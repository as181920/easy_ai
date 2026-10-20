module DecisionScaling
  module_function

  def memory_choices(root)
    results = Dir[File.join(root, "preflight/*/result.json")].sort.map do |path|
      value = JSON.parse(File.read(path))
      manifest = JSON.parse(File.read(File.join(File.dirname(path), "initialization.json")))
      value.merge("profile" => manifest.fetch("profile"), "scenario" => manifest.fetch("scenario", "full"), "path" => path)
    end
    full = results.select { |value| value.fetch("passed") && value.fetch("scenario") == "full" }
    raise "Base full-shape preflight has not passed" unless full.any? { |value| value.fetch("profile") == "base" }
    profile = PROFILES.keys.reverse.find { |name| full.any? { |value| value.fetch("profile") == name } }
    observed = JSON.parse(File.read(File.join(root, "data-audit.json"))).fetch("token_limits_observed")
    controlled = results.select do |value|
      manifest = JSON.parse(File.read(File.join(File.dirname(value.fetch("path")), "initialization.json")))
      value.fetch("passed") && value.fetch("scenario") == "controlled" && observed.all? { |key, limit| manifest.fetch("probe_lengths").fetch(key) >= limit }
    end
    names = ["base", profile].uniq
    common = names.map { |name| controlled.select { |value| value.fetch("profile") == name }.map { |value| value.fetch("microbatch") } }.reduce(:&)
    raise "No common controlled microbatch passed" if common.empty?
    { "profile" => profile, "controlled_microbatch" => common.max,
      "full_microbatch" => names.to_h { |name| [name, full.select { |value| value.fetch("profile") == name }.map { |value| value.fetch("microbatch") }.max] },
      "results_sha256" => results.to_h { |value| [value.fetch("path").delete_prefix(root + "/"), sha(value.fetch("path"))] } }
  end

  def prepare(root)
    path = File.join(root, "protocol.json")
    raise "Scaling protocol already frozen" if File.exist?(path)
    hardware = memory_choices(root)
    profiles = ["base", hardware.fetch("profile")].uniq
    files = Dir[File.join(root, "data/*.jsonl")].sort
    raise "Scaling panels incomplete" unless files.size == 15
    cross_path = File.join(root, "cross-round-audit.json")
    cross = JSON.parse(File.read(cross_path))
    raise "Cross-round independence has not passed" unless cross.fetch("passed")
    cross.fetch("panels_sha256").each { |name, digest| raise "Cross-audited data changed" unless sha(File.join(root, "data", "#{name}.jsonl")) == digest }
    source = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort
    source.each do |file|
      destination = File.join(root, "source-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    value = { "version" => 6, "created_at" => Time.now.utc.iso8601, "profiles" => profiles, "data_scales" => [32, 512],
      "hardware" => hardware, "config_by_profile" => profiles.to_h { |profile| [profile, model_config(loaded.fetch(:model), profile, microbatch: hardware.fetch("controlled_microbatch")).to_h] },
      "parent_sha256" => loaded.fetch(:weights_fingerprint), "tokenizer_fingerprint" => loaded.fetch(:tokenizer).fingerprint,
      "files_sha256" => files.to_h { |file| [File.basename(file), sha(file)] }, "data_audit_sha256" => sha(File.join(root, "data-audit.json")),
      "cross_round_audit_sha256" => sha(cross_path),
      "plan_sha256" => sha("docs/decision/v06-plan.md"), "source_sha256" => source.to_h { |file| [file, sha(file)] },
      "screen_steps" => 1500, "extended_steps" => 4000, "evaluation_every" => 250, "update_ceiling" => UPDATE_LIMIT,
      "mastery" => { "known_balanced" => 0.95, "dev_balanced" => 0.55, "mixed_groups" => 0.9, "stable_checks" => 2 },
      "publication" => { "binary_balanced" => 0.75, "joint_known_recall" => 0.7, "joint_unknown_recall" => 0.7, "mixed_groups" => 0.5, "parent_gain" => 0.05, "natural_minimum_accuracy" => 0.65, "natural_tolerance" => 0.03 },
      "required_development" => %w[canonical seen_combinations novel_actor], "diagnostic_development" => %w[novel_event],
      "sampler" => QuadTrainer::SIGNATURE,
      "scope" => "Frozen two-factor controlled screen. A controlled checkpoint is not a factual delivery: reviewed natural development, full-recipe confirmation and separate calibration/acceptance remain required." }
    FileUtils.cp("docs/decision/v06-plan.md", File.join(root, "plan-frozen.md"))
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts "Frozen v06 factorial: #{profiles} x 32/512 worlds; common controlled microbatch #{hardware.fetch('controlled_microbatch')}"
  end

  def verify(root)
    path = File.join(root, "protocol.json")
    raise "Scaling protocol changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    value = JSON.parse(File.read(path))
    raise "Parent changed" unless sha(File.join(PARENT, "weights.pt")) == value.fetch("parent_sha256")
    value.fetch("files_sha256").each { |name, digest| raise "Scaling data changed: #{name}" unless sha(File.join(root, "data", name)) == digest }
    raise "Data audit changed" unless sha(File.join(root, "data-audit.json")) == value.fetch("data_audit_sha256")
    raise "Cross-round audit changed" unless sha(File.join(root, "cross-round-audit.json")) == value.fetch("cross_round_audit_sha256")
    value.fetch("hardware").fetch("results_sha256").each { |name, digest| raise "Hardware trial changed" unless sha(File.join(root, name)) == digest }
    value
  end
end
