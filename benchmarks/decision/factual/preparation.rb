module DecisionFactual
  module_function

  def prepare(root)
    raise ArgumentError, "Output exists; preserve prior evidence" if File.exist?(root)
    FileUtils.mkdir_p(File.join(root, "data"))
    historical, _, history_files = DecisionRelease.history
    loaded = EasyAI::Decision::Checkpoint.load(INITIALIZERS.fetch("broad"))
    tokenizer = loaded.fetch(:tokenizer)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config(1337))
    raw = EasyAI::Decision::Data::SemanticAdapter.each("data/decision/downloads/semantics").to_a
    grouped = EasyAI::Decision::Data::SemanticCorpus.new(raw).split_rows
    counts = Hash.new(0)
    historical_groups = historical_components(grouped, historical)
    fresh = grouped.select { |_, group, split| split == "test" && !historical_groups.include?(group) }
    # Entire source components are assigned before caps/length filtering, without targets.
    development = fresh.group_by { |_, group, _| digest("v02-natural:#{group}").to_i(16) % 5 }
    natural = %w[validation calibration test].each_with_index.to_h do |split, bucket|
      eligible = bucket == 2 ? development.values_at(2, 3, 4).compact.flatten(1) : development.fetch(bucket, [])
      converted = eligible.map { |row, group, _| convert(row, group) }
      [split, select_supported(converted, per_source: split == "test" ? 200 : 64, collator: collator, counts: counts)]
    end
    controlled = EasyAI::Decision::Data::FactualContrasts.splits
    train = grouped.select { |_, _, split| split == "train" }.map { |row, group, _| convert(row, group) }
    train = select_supported(train, per_source: 4000, collator: collator, counts: counts)
    replay = raw_rows("runs/decision/release-v01-corrective/data/train.jsonl")
      .group_by { |row| row.fetch("language") }.values.flat_map { |rows| stable(rows, "v02-replay").first(2000) }
    panels = %w[validation calibration test].to_h do |split|
      routing = split == "test" ? raw_rows("runs/decision/release-v01-corrective/data/test.jsonl") :
        raw_rows("runs/decision/release-v01-corrective/data/#{split}.jsonl").group_by { |row| row.fetch("language") }.values.flat_map { |rows| stable(rows, "v02-routing-#{split}").first(64) }
      [split, natural.fetch(split) + controlled.fetch(split) + routing]
    end
    panels["train"] = train + controlled.fetch("train") + replay
    panels.each do |name, rows|
      rows.each { |row| EasyAI::Decision::Data::Example.new(row) }
      write_rows(File.join(root, "data/#{name}.jsonl"), rows)
    end
    datasets = %w[train validation calibration test].map { |split| EasyAI::Decision::Data::Dataset.new(File.join(root, "data/#{split}.jsonl")) }
    EasyAI::Decision::Data::Dataset.assert_disjoint!(*datasets)
    sets = panels.values.map { |rows| rows.map { |row| material(row.fetch("state")) }.to_set }
    sets.combination(2).each { |left, right| raise "Material leakage" unless (left & right).empty? }
    raise "Training budget outside pilot cap" unless (20_000..40_000).cover?(panels.fetch("train").size)
    natural.each do |split, rows|
      raise "Insufficient natural panel: #{split}" unless rows.group_by { |row| row.fetch("source") }.values.all? { |items| items.size >= 50 } && rows.map { |row| row.fetch("source") }.uniq.size == 3
    end
    tokenizer.save(File.join(root, "data/tokenizer.json"))
    fit = fitting_rows(panels.fetch("train"))
    write_rows(File.join(root, "data/fit.jsonl"), fit)
    protocol = { "version" => 2, "status" => "prepared", "created_at" => Time.now.utc.iso8601,
      "initializers" => INITIALIZERS, "steps" => STEPS, "arms" => ARMS, "seed" => 1337,
      "fit" => { "maximum_steps" => 1000, "evaluation_every" => 100, "stable_passing_checks" => 2 },
      "margin_weight" => 0.2, "margin" => 1.0, "config" => config(1337).to_h,
      "files_sha256" => Dir.glob(File.join(root, "data/*")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
      "source_sha256" => history_files.merge(Dir.glob("data/decision/downloads/semantics/*").select { |p| File.file?(p) }.to_h { |p| [p, Digest::SHA256.file(p).hexdigest] }),
      "counts" => panels.transform_values { |rows| rows.group_by { |row| "#{row.fetch('source')}/#{row.fetch('language')}" }.transform_values(&:size) },
      "natural_length_audit" => counts, "natural_test_fresh" => true, "routing_test_scope" => "Previously observed regression panel, not fresh acceptance",
      "selection" => { "routing_tolerance" => 0.03, "reference" => "v0.1 on identical validation",
        "primary" => "Equal-language mean of within-language source-macro factual accuracy", "tie_break" => "factual NLL" },
      "confirmation" => { "minimum_validation_gain" => 0.03, "seed" => 2027 },
      "publication" => { "minimum_factual_language_gain" => 0.03, "minimum_pair_accuracy" => 0.8,
        "minimum_natural_language_gain" => 0.0, "maximum_routing_regression" => 0.03, "confirmation_required" => true },
      "scope" => "Own weights; one shared bilingual model; CE versus CE+signed paired margin. Eight trained event families, four withheld. Official unused QA/NLI dev components reserved before fitting; all historical prepared material excluded. Routing is regression-only. No teacher, RL, growth or business integration." }
    write(File.join(root, "protocol.json"), protocol)
    puts JSON.pretty_generate(protocol.slice("counts", "natural_length_audit"))
  end

  def fitting_rows(rows)
    controlled = rows.select { |row| row.fetch("source").start_with?("Factual-") }
      .group_by { |row| row.fetch("world").fetch("event") }.values.each_with_index.flat_map do |items, index|
        group = items.first.fetch("group_id")
        items.select { |row| row.fetch("group_id") == group && [index % 4, 4].include?(row.fetch("world").fetch("variant")) }
      end
    natural = rows.reject { |row| row.fetch("source").start_with?("Factual-") || row.fetch("source") == "MASSIVE-Scenario" }
      .group_by { |row| row.fetch("language") }.values.flat_map { |items| stable(items, "v02-fit").first(32) }
    routing = rows.select { |row| row.fetch("source") == "MASSIVE-Scenario" }.group_by { |row| row.fetch("language") }
      .values.flat_map { |items| stable(items, "v02-fit").first(8) }
    controlled + natural + routing
  end

  def convert(row, group)
    options = row.fetch(:options).map { |id, text| { "id" => id, "text" => text } }
    { "id" => digest([group, row.fetch(:state), row.fetch(:question)].join("\n")), "group_id" => group,
      "language" => row.fetch(:language), "source" => row.fetch(:source), "partition" => row.fetch(:partition),
      "state" => row.fetch(:state), "question" => row.fetch(:question), "options" => options, "target" => row.fetch(:target) }
  end

  def select_supported(rows, per_source:, collator:, counts:)
    rows.group_by { |row| row.fetch("source") }.values.flat_map do |source_rows|
      groups = stable(source_rows, "v02-source").group_by { |row| row.fetch("group_id") }
      selected = []
      groups.each_value do |items|
        break if selected.size >= per_source
        items.each do |row|
          break if selected.size >= per_source
          begin
            collator.state_tokens(row.fetch("state"))
            row.fetch("options").each { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }
            selected << row
          rescue ArgumentError => error
            raise unless error.message.include?("exceeds")
            counts["#{row.fetch('source')}/over_length"] += 1
          end
        end
      end
      selected.uniq { |row| row.fetch("id") }
    end
  end

  def stable(rows, salt)
    rows.sort_by { |row| digest("#{salt}:#{row.fetch('id')}") }
  end

  def material(text)
    EasyAI::Decision::Data::NaturalCorpus.material(text)
  end

  def digest(text)
    Digest::SHA256.hexdigest(text)
  end
end
