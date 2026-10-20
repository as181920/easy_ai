module DecisionScaling
  COUNTS = { "train" => 512, "validation" => 32, "calibration" => 32, "test" => 128 }.freeze
  HOLDOUT_COUNTS = { "train" => 0, "validation" => 16, "calibration" => 16, "test" => 32 }.freeze
  DATA_SCALES = [32, 128, 512].freeze
  module_function

  def controlled_rows(split, count:, panel: "canonical")
    allocation = panel.to_s == "canonical" ? COUNTS : HOLDOUT_COUNTS
    count.times.flat_map do |index|
      %w[en-US zh-CN].flat_map do |language|
        EasyAI::Decision::Data::ScalableJudgmentCorpus.new(split, index, language, counts: allocation, panel: panel).rows
      end
    end
  end

  def binary_rows(raw)
    raw.select { |row| row.dig("world", "profile") == "binary" }
  end

  def prepare_data(root)
    directory = File.join(root, "data")
    raise "Preserve prepared scaling data" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    panels = COUNTS.to_h { |split, count| [split, controlled_rows(split, count: count)] }
    %w[novel_actor novel_event].each do |panel|
      %w[validation calibration test].each do |split|
        panels["#{split}-#{panel}"] = controlled_rows(split, count: HOLDOUT_COUNTS.fetch(split), panel: panel)
      end
    end
    panels.each do |name, rows|
      EasyAI::Decision::Data::QualityAudit.integrity!(rows)
      DecisionRobust.write_rows(File.join(directory, "#{name}.jsonl"), rows)
    end
    # Different evaluation slices also remain mutually disjoint, including full
    # material across languages and all training orders/truth assignments.
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(*panels.values)
    DATA_SCALES.each do |scale|
      rows = panels.fetch("train").select { |row| row.fetch("group_id").split(":").last.to_i < scale }
      DecisionRobust.write_rows(File.join(directory, "train-#{scale}-binary.jsonl"), binary_rows(rows))
    end
    probe = panels.fetch("train").select { |row| row.fetch("group_id").split(":").last.to_i < 16 }
    DecisionRobust.write_rows(File.join(directory, "probe-binary.jsonl"), binary_rows(probe))
    tokenizer = EasyAI::Decision::Checkpoint.load(PARENT).fetch(:tokenizer)
    config = EasyAI::Decision::Checkpoint.load(PARENT).fetch(:model).config
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config)
    token_limits = { "state" => 0, "option" => 0 }
    panels.each_value do |raw|
      raw.each do |row|
        token_limits["state"] = [token_limits.fetch("state"), collator.state_tokens(row.fetch("state")).size].max
        row.fetch("options").each do |option|
          length = collator.option_tokens(row.fetch("question"), option.fetch("text")).size
          token_limits["option"] = [token_limits.fetch("option"), length].max
        end
      end
    end
    stats = panels.transform_values do |raw|
      atoms = raw.flat_map { |row| row.fetch("world").fetch("facts").map { |fact| [row.fetch("language"), fact.fetch("actor")] } }.uniq
      roles = raw.select { |row| row.dig("world", "profile") == "joint" }.group_by { |row| [row.fetch("language"), row.dig("world", "actor")] }
      { "rows" => raw.size, "families_per_language" => raw.group_by { |row| row.fetch("language") }.transform_values { |items| items.map { |row| row.fetch("group_id") }.uniq.size },
        "actor_identities" => atoms.size, "unique_inputs" => raw.map { |row| [row.fetch("state"), row.fetch("question"), row.fetch("options")] }.uniq.size,
        "unknown_only_actor_identities" => roles.values.count { |items| items.none? { |row| row.dig("world", "kind") == "known" } },
        "known_only_actor_identities" => roles.values.count { |items| items.none? { |row| row.dig("world", "kind") == "absent_actor" } },
        "by_language_target" => raw.group_by { |row| "#{row.fetch('language')}/#{row.fetch('target')}" }.transform_values(&:size) }
    end
    write(File.join(root, "data-audit.json"), { "panels" => stats, "token_limits_observed" => token_limits, "truncations" => collator.truncated,
      "split_counts" => COUNTS, "offsets" => EasyAI::Decision::Data::JudgmentCorpus.split_offsets(COUNTS),
      "scope" => "Integrity, complete independent truth assignments, exact family/material disjointness and token limits; no semantic checkpoint evaluation or acceptance opened." })
    puts "Prepared scale data: #{stats.transform_values { |cell| cell.fetch('rows') }}"
  end

  def seen_combination_rows(training_rows, count: 32, reserved_worlds: COUNTS.values.sum)
    raise ArgumentError, "Seen-combination count must be positive and even" unless count.is_a?(Integer) && count.positive? && count.even?
    corpus = EasyAI::Decision::Data::ScalableJudgmentCorpus
    %w[en-US zh-CN].flat_map do |language|
      known = training_rows.select { |row| row.fetch("language") == language }.flat_map { |row| row.fetch("world").fetch("facts").map { |fact| fact.fetch("actor") } }.to_set
      [0, 1].flat_map do |axis|
        slots = corpus.catalog(language, "canonical", axis).each_index.select do |slot|
          actors, = corpus.catalog(language, "canonical", axis).fetch(slot)
          slot >= (reserved_worlds + 1) / 2 && actors.first(axis.zero? ? 2 : 1).all? { |name| known.include?(name) }
        end.first(count / 2)
        raise "Not enough disjoint seen-name combinations" unless slots.size == count / 2
        slots.each_with_index.flat_map do |slot, index|
          generator = corpus.new(:validation, 0, language, counts: { train: slot * 2 + axis, validation: 1 })
          generator.instance_variable_set(:@group, "scale-v06:seen_combinations:validation:#{axis}:#{index}")
          generator.rows
        end
      end
    end
  end

  def prepare_seen(root)
    directory = File.join(root, "data")
    path = File.join(directory, "validation-seen_combinations.jsonl")
    raise "Seen-combination panel exists" if File.exist?(path)
    training = DecisionRelational.rows(File.join(directory, "train-32-binary.jsonl"))
    rows = seen_combination_rows(training)
    audit_path = File.join(root, "data-audit.json")
    audit = JSON.parse(File.read(audit_path))
    panels = audit.fetch("panels").keys.map { |name| DecisionRelational.rows(File.join(directory, "#{name}.jsonl")) }
    EasyAI::Decision::Data::QualityAudit.integrity!(rows)
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(rows, *panels)
    DecisionRobust.write_rows(path, rows)
    audit.fetch("panels")["validation-seen_combinations"] = { "rows" => rows.size, "families_per_language" => { "en-US" => 32, "zh-CN" => 32 },
      "scope" => "Every recorded actor appears in the smallest training subset; facts use catalog slots outside all canonical train/dev/cal/test reservations. Constructed from training atoms before training, without inspecting predictions." }
    write(audit_path, audit)
    puts "Prepared #{rows.size} seen-name, unseen-combination development rows"
  end

  def audit_limits(root)
    path = File.join(root, "data-audit.json")
    audit = JSON.parse(File.read(path))
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config)
    maxima = { "state" => 0, "option" => 0 }
    audit.fetch("panels").each_key do |name|
      File.foreach(File.join(root, "data", "#{name}.jsonl")) do |line|
        row = JSON.parse(line)
        maxima["state"] = [maxima.fetch("state"), collator.state_tokens(row.fetch("state")).size].max
        row.fetch("options").each { |option| maxima["option"] = [maxima.fetch("option"), collator.option_tokens(row.fetch("question"), option.fetch("text")).size].max }
      end
    end
    audit["token_limits_observed"], audit["truncations"] = maxima, collator.truncated
    write(path, audit)
    puts "All frozen panels: observed maxima #{maxima}, truncations=#{collator.truncated}"
  end

  def cross_round_audit(root)
    path = File.join(root, "cross-round-audit.json")
    raise "Cross-round audit already frozen" if File.exist?(path)
    prior = Dir["runs/decision/**/data/*.jsonl"].reject { |file| File.expand_path(file).start_with?(File.expand_path(root) + "/") }.sort
    manifests = prior.to_h { |file| [file, sha(file)] }
    states = Set.new
    manifests.group_by { |_, digest| digest }.each_value do |copies|
      File.foreach(copies.first.first) do |line|
        row = JSON.parse(line)
        states << EasyAI::Decision::Data::QualityAudit.digest(row.fetch("state")) if row["state"]
      end
    end
    audit = JSON.parse(File.read(File.join(root, "data-audit.json")))
    overlap = audit.fetch("panels").keys.to_h do |name|
      matches = Set.new
      File.foreach(File.join(root, "data", "#{name}.jsonl")) do |line|
        record = JSON.parse(line).fetch("state")
        digest = EasyAI::Decision::Data::QualityAudit.digest(record)
        matches << digest if states.include?(digest)
      end
      [name, matches.size]
    end
    value = { "old_data_sha256" => manifests, "old_unique_record_materials" => states.size,
      "panels_sha256" => audit.fetch("panels").keys.to_h { |name| [name, sha(File.join(root, "data", "#{name}.jsonl"))] },
      "overlapping_unique_record_materials" => overlap, "passed" => overlap.values.all?(&:zero?),
      "scope" => "Conservative literal record exclusion against all previous local Decision data panels, including reserved panels. New controlled actor identities also differ from the legacy generator pool. This is not a claim that individual words were absent from parent/tokenizer training." }
    write(path, value)
    raise "Cross-round material overlap: #{overlap}" unless value.fetch("passed")
    puts "Cross-round audit passed against #{prior.size} old panels / #{states.size} unique record materials"
  end
end
