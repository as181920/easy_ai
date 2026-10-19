module DecisionRelational
  module_function

  def prepare(root)
    raise ArgumentError, "Preserve existing experiment" if File.exist?(root)
    DecisionJudgment.verify(SOURCE)
    parent = EasyAI::Decision::Checkpoint.load(PARENT)
    config = parent.fetch(:model).config.with(model: { dropout: 0.0 }, training: {
      learning_rate: 0.0003, warmup_steps: 0, steps: 1000, eval_every: 100, checkpoint_every: 100,
      early_stopping_patience: 0, balance_labels: false, balance_sources: false, paired_sampling: false,
      resample_negatives: false, track_coverage: true, choice_microbatch: 4, gradient_accumulation: 8
    })
    files = %w[fit.jsonl validation.jsonl candidate-train.jsonl candidate-probe.jsonl]
    panels = files.to_h { |name| [name, rows(File.join(SOURCE, "data", name))] }
    raise "Unexpected fitting size" unless panels.fetch("fit.jsonl").size == 128
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(panels.fetch("candidate-train.jsonl"), panels.fetch("validation.jsonl"))
    FileUtils.mkdir_p(File.join(root, "data"))
    files.each { |name| FileUtils.cp(File.join(SOURCE, "data", name), File.join(root, "data", name)) }
    protocol = { "version" => 5, "created_at" => Time.now.utc.iso8601, "source" => SOURCE,
      "parent_weights_sha256" => parent.fetch(:weights_fingerprint), "tokenizer_fingerprint" => parent.fetch(:tokenizer).fingerprint,
      "files_sha256" => files.to_h { |name| [name, sha(File.join(root, "data", name))] },
      "configs" => ARMS.to_h { |arm| [arm, config.with(model: { encoding_mode: arm }).to_h] },
      "source_revision" => IO.popen(%w[git rev-parse HEAD], &:read).strip,
      "source_sha256" => Dir.glob("{lib/easy_ai,benchmarks/decision}/**/*.rb").sort.to_h { |path| [path, sha(path)] },
      "fit" => { "maximum_steps" => 1000, "balanced" => 0.99, "mixed_groups" => 0.9, "stable_checks" => 2 },
      "scope" => "Reviewed v0.4 training fit; validation is observed development, not fresh acceptance. No acceptance predictions authorized by a fitting result. Fresh optimizers, own v0.1 parent, sinusoidal positions; joint omits separate interaction and matching modules." }
    write(File.join(root, "protocol.json"), protocol)
    write(File.join(root, "protocol-sha256.json"), { "sha256" => sha(File.join(root, "protocol.json")) })
    puts "Prepared v0.5: #{panels.transform_values(&:size)}; acceptance remains untouched"
  end

  def verify(root)
    frozen = JSON.parse(File.read(File.join(root, "protocol-sha256.json"))).fetch("sha256")
    raise "Protocol changed" unless sha(File.join(root, "protocol.json")) == frozen
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    protocol.fetch("files_sha256").each do |name, digest|
      raise "Frozen data changed: #{name}" unless sha(File.join(root, "data", name)) == digest
    end
    raise "Parent changed" unless sha(File.join(PARENT, "weights.pt")) == protocol.fetch("parent_weights_sha256")
    protocol
  end

  def mastered?(value, rules)
    value.fetch("by_language").keys.sort == %w[en-US zh-CN] && value.fetch("by_language").values.all? do |cell|
      cell.fetch("balanced") && cell.fetch("balanced") >= rules.fetch("balanced") && %w[actor event].all? do |kind|
        cell.fetch("#{kind}_binding_groups").positive? && cell.fetch("#{kind}_binding_all_correct") >= rules.fetch("mixed_groups")
      end
    end
  end
end
