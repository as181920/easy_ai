module DecisionScaling
  module_function

  # This exports labels for a documented target-blind review, not automatic gold.
  # It is independent of model predictions and does not authorize supervision.
  def prepare_natural_review(root)
    verify(root)
    directory = File.join(root, "natural-review")
    raise "Preserve natural review reservation" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    old = Dir["{runs,data}/decision/**/*.jsonl"].reject { |path| File.expand_path(path).start_with?(root + "/") }.sort
    blocked = Set.new
    old_hashes = {}
    seen_files = Set.new
    old.each do |path|
      digest = sha(path)
      old_hashes[path] = digest
      next unless seen_files.add?(digest)
      File.foreach(path) do |line|
        row = JSON.parse(line)
        blocked << EasyAI::Decision::Data::QualityAudit.digest(row.fetch("state")) if row["state"].is_a?(String)
      end
    end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config.with(input: { truncation: "error" }))
    candidates, exclusions = [], Hash.new(0)
    material_seen = Set.new
    sources = "data/decision/downloads/semantics"
    EasyAI::Decision::Data::SemanticAdapter.each(sources) do |row|
      next unless %w[BoolQ OCNLI].include?(row.fetch(:source))
      material = EasyAI::Decision::Data::QualityAudit.digest(row.fetch(:state))
      reason = if blocked.include?(material)
        "historical_material"
               elsif !material_seen.add?(material)
        "repeated_material"
               elsif collator.state_tokens(row.fetch(:state)).size > 256 || row.fetch(:options).values.any? { |text| collator.option_tokens(row.fetch(:question), text).size > 64 }
        "outside_model_limits"
               end
      if reason
        exclusions[reason] += 1
        next
      end
      id = "scale-v06:natural:#{material}"
      candidates << { "id" => id, "group_id" => id, "source" => row.fetch(:source), "language" => row.fetch(:language),
        "state" => row.fetch(:state), "question" => row.fetch(:question), "target" => row.fetch(:target),
        "options" => row.fetch(:options).map { |key, text| { "id" => key, "text" => text } }, "source_partition" => row.fetch(:partition) }
    rescue ArgumentError => error
      raise unless error.message.include?("exceeds")
      exclusions["outside_model_limits"] += 1
    end
    buckets = candidates.group_by { |row| [row.fetch("language"), row.fetch("target")] }.transform_values do |rows|
      rows.sort_by { |row| EasyAI::Decision::Data::QualityAudit.digest("scale-v06:natural:#{row.fetch('id')}") }
    end
    write(File.join(directory, "inventory.json"), { "candidates" => buckets.to_h { |key, rows| [key.join("/"), rows.size] }, "exclusions" => exclusions })
    allocation = { "train" => { "en-US" => [100, 100, 0], "zh-CN" => [152, 16, 32] },
      "development" => { "en-US" => [16, 16, 0], "zh-CN" => [12, 12, 8] },
      "calibration" => { "en-US" => [16, 16, 0], "zh-CN" => [16, 8, 8] },
      "test" => { "en-US" => [32, 32, 0], "zh-CN" => [20, 20, 24] } }
    panels = allocation.to_h do |split, languages|
      selected = languages.flat_map do |language, counts|
        %w[yes no unknown].zip(counts).flat_map do |target, count|
          rows = buckets.fetch([language, target], []).shift(count)
          raise "Insufficient fresh #{language}/#{target}: #{split} requires #{count}" unless rows.size == count
          rows.map { |row| row.merge("review_id" => EasyAI::Decision::Data::QualityAudit.digest("review:#{row.fetch('id')}")[0, 16]) }
        end
      end.sort_by { |row| EasyAI::Decision::Data::QualityAudit.digest("blind:#{row.fetch('review_id')}") }
      EasyAI::Decision::Data::QualityAudit.integrity!(selected)
      DecisionRobust.write_rows(File.join(directory, "#{split}-selected.jsonl"), selected)
      blind = selected.map { |row| row.slice("review_id", "language", "state", "question", "options").merge("contract" => row.fetch("source")) }
      DecisionRobust.write_rows(File.join(directory, "#{split}-blind.jsonl"), blind)
      [split, selected]
    end
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(*panels.values)
    paths = Dir[File.join(directory, "*.jsonl")]
    value = { "allocation_before_review" => allocation, "excluded" => exclusions, "historical_files_sha256" => old_hashes,
      "quota_limitation" => "The conservative first-encountered-candidate-per-record inventory contains 56 fresh Chinese contradiction examples after history/length filtering. This is not exhaustive availability across alternative hypotheses for the same records. Reserve balanced known development/test; training is class-skewed and needs explicit label-balanced sampling. Counts precede blind review.",
      "source_files_sha256" => Dir[File.join(sources, "{boolq.zip,ocnli-*.jsonl}")].sort.to_h { |path| [path, sha(path)] },
      "files_sha256" => paths.to_h { |path| [File.basename(path), sha(path)] },
      "scope" => "Fresh literal record materials excluded against prior local raw/derived panels. Original source labels are unqualified until target-blind review. Supported natural scope: state <=256 and question/option <=64 own-tokenizer tokens. One material per selected example. No model predictions used; no factual delivery established." }
    write(File.join(directory, "reservation.json"), value)
    write(File.join(directory, "reservation.sha256.json"), { "sha256" => sha(File.join(directory, "reservation.json")) })
    puts "Reserved target-blind natural panels: #{panels.transform_values(&:size)}"
  end

  def order_natural_review(root)
    directory = File.join(root, "natural-review")
    path = File.join(directory, "reservation.json")
    raise "Natural reservation changed" unless sha(path) == JSON.parse(File.read(File.join(directory, "reservation.sha256.json"))).fetch("sha256")
    reservation = JSON.parse(File.read(path))
    reservation.fetch("files_sha256").each { |name, digest| raise "Natural export changed" unless sha(File.join(directory, name)) == digest }
    ordered = File.join(directory, "ordered")
    raise "Preserve ordered blind exports" if File.exist?(ordered)
    FileUtils.mkdir_p(ordered)
    %w[train development calibration test].each do |split|
      rows = DecisionRelational.rows(File.join(directory, "#{split}-selected.jsonl")).sort_by { |row| EasyAI::Decision::Data::QualityAudit.digest("blind:#{row.fetch('review_id')}") }
      blind = rows.map { |row| row.slice("review_id", "language", "state", "question", "options").merge("contract" => row.fetch("source")) }
      DecisionRobust.write_rows(File.join(ordered, "#{split}-selected.jsonl"), rows)
      DecisionRobust.write_rows(File.join(ordered, "#{split}-blind.jsonl"), blind)
    end
    write(File.join(ordered, "manifest.json"), { "reservation_sha256" => sha(path),
      "files_sha256" => Dir[File.join(ordered, "*.jsonl")].sort.to_h { |file| [File.basename(file), sha(file)] },
      "scope" => "Normalize review order by review-ID hash before any assistant manual review. Original reservation remains unchanged; class/language reservation order is not presented to the reviewer." })
    write(File.join(ordered, "manifest.sha256.json"), { "sha256" => sha(File.join(ordered, "manifest.json")) })
  end

  def import_natural_review(root, split)
    raise ArgumentError, "Only training/development reviews precede qualification" unless %w[train development].include?(split.to_s)
    directory = File.join(root, "natural-review")
    manifest_path = File.join(directory, "ordered/manifest.json")
    raise "Ordered review manifest changed" unless sha(manifest_path) == JSON.parse(File.read(File.join(directory, "ordered/manifest.sha256.json"))).fetch("sha256")
    manifest = JSON.parse(File.read(manifest_path))
    raise "Natural reservation changed" unless sha(File.join(directory, "reservation.json")) == manifest.fetch("reservation_sha256")
    manifest.fetch("files_sha256").each { |name, digest| raise "Blind review export changed" unless sha(File.join(directory, "ordered", name)) == digest }
    path = File.join(directory, "#{split}-audit.json")
    raise "Preserve imported natural review" if File.exist?(path)
    selected = DecisionRelational.rows(File.join(directory, "ordered", "#{split}-selected.jsonl"))
    review_path = File.join(directory, "reviews", "#{split}.jsonl")
    audit = EasyAI::Decision::Data::QualityAudit.adjudicate(selected, DecisionRelational.rows(review_path))
    accepted = audit.fetch("decisions").select { |row| row.fetch("agreement") && !row.fetch("excluded") }
    audit["qualified_counts"] = accepted.group_by { |row| "#{row.fetch('language')}/#{row.fetch('target')}" }.transform_values(&:size)
    audit["review_sha256"] = sha(review_path)
    audit["export_manifest_sha256"] = sha(manifest_path)
    audit["reviewer_scope"] = "Codex assistant, one target-blind review; not independent human or second-model agreement. Source disagreements remain excluded, not automatically relabelled. Review happens before supervised recipe selection."
    write(path, audit)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts JSON.pretty_generate(audit.slice("by_source", "qualified_counts", "reviewer_scope"))
  end
end
