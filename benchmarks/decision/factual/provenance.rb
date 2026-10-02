module DecisionFactual
  module_function

  def historical_components(grouped, known_material)
    grouped.filter_map { |row, group, _| group if known_material.include?(material(row.fetch(:state))) }.to_set
  end

  def historical_material(protocol)
    protocol.fetch("source_sha256").each_with_object(Set.new) do |(path, sha), known|
      next unless path.end_with?(".jsonl") && !path.include?("/downloads/")
      raise "Historical source changed: #{path}" unless Digest::SHA256.file(path).hexdigest == sha
      File.foreach(path) do |line|
        next if line.strip.empty?
        row = JSON.parse(line)
        text = row["state"] || row["text"]
        known << material(text) if text
      end
    end
  end

  def audit(root)
    protocol = verify(root)
    raise "Final evaluation already opened" if File.exist?(File.join(root, "acceptance-opened.json"))
    raise "Provenance already frozen" if File.exist?(File.join(root, "provenance.json"))
    historical = historical_material(protocol)
    protocol.fetch("source_sha256").each do |path, sha|
      next unless path.include?("/downloads/")
      raise "Public cache changed: #{path}" unless Digest::SHA256.file(path).hexdigest == sha
    end
    raw = EasyAI::Decision::Data::SemanticAdapter.each("data/decision/downloads/semantics").to_a
    grouped = EasyAI::Decision::Data::SemanticCorpus.new(raw).split_rows
    known = historical_components(grouped, historical)
    counts = {}
    files = { "calibration" => "calibration-clean.jsonl", "test" => "acceptance.jsonl" }
    files.each do |split, name|
      rows = raw_rows(File.join(root, "data/#{split}.jsonl"))
      excluded, kept = rows.partition { |row| known.include?(row.fetch("group_id")) }
      counts[split] = { "excluded_rows" => excluded.size, "excluded_groups" => excluded.map { |row| row.fetch("group_id") }.uniq,
        "retained_rows" => kept.size, "by_source" => kept.group_by { |row| row.fetch("source") }.transform_values(&:size) }
      natural = kept.reject { |row| row.fetch("source").start_with?("Factual-") || row.fetch("source") == "MASSIVE-Scenario" }
      sources = natural.group_by { |row| row.fetch("source") }
      raise "Insufficient clean natural panel" unless sources.size == 3 && sources.values.all? { |items| items.size >= 50 }
      write_rows(File.join(root, "data", name), kept)
    end
    write(File.join(root, "provenance.json"), { "created_at" => Time.now.utc.iso8601, "panels" => counts,
      "files_sha256" => files.values.to_h { |name| [name, Digest::SHA256.file(File.join(root, "data", name)).hexdigest] },
      "original_protocol_sha256" => Digest::SHA256.file(File.join(root, "protocol.json")).hexdigest,
      "scope" => "Whole historically connected public components excluded without model predictions or label-dependent choices. Original training/validation remain frozen; clean calibration and acceptance fixed before temperatures and final evaluation." })
    puts JSON.pretty_generate(counts)
  end

  def verify_panels(root)
    provenance = JSON.parse(File.read(File.join(root, "provenance.json")))
    raise "Original protocol changed" unless Digest::SHA256.file(File.join(root, "protocol.json")).hexdigest == provenance.fetch("original_protocol_sha256")
    provenance.fetch("files_sha256").each do |name, sha|
      raise "Clean panel changed: #{name}" unless Digest::SHA256.file(File.join(root, "data", name)).hexdigest == sha
    end
    provenance
  end
end
