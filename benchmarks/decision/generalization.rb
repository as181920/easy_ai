#!/usr/bin/env ruby
require_relative "evidence_evaluation" unless defined?(EvidenceEvaluation)
require "zlib"
require "rubygems/package"

# Supplemental zero-shot panel, added before any final test scores were opened.
# Candidate descriptions define output classes; they do not classify input text.
module DecisionGeneralization
  ROOT = EvidenceExperiment::ROOT
  LABELS = {
    "AG-News" => ["World news", "Sports news", "Business news", "Science and technology news"],
    "Emotion" => %w[sadness joy love anger fear surprise]
  }.freeze
  SCENARIOS = {
    "alarm" => ["闹钟", "alarms"], "audio" => ["音频设置", "audio settings"],
    "calendar" => ["日历与预约", "calendar and appointments"], "cooking" => ["烹饪", "cooking"],
    "datetime" => ["日期与时间", "date and time"], "email" => ["电子邮件", "email"],
    "general" => ["一般对话", "general conversation"], "iot" => ["智能家居设备", "smart home devices"],
    "lists" => ["清单", "lists"], "music" => ["音乐信息与管理", "music information and management"],
    "news" => ["新闻", "news"], "play" => ["播放媒体", "media playback"],
    "qa" => ["知识问答", "factual questions and answers"], "recommendation" => ["推荐建议", "recommendations"],
    "social" => ["社交媒体", "social media"], "takeaway" => ["外卖", "takeaway food"],
    "transport" => ["交通出行", "transport"], "weather" => ["天气", "weather"]
  }.freeze
  DOWNLOADS = File.join(ROOT, "data/decision/downloads/generalization")
  module_function

  def download
    %w[ag_news emotion].each do |name|
      dataset, config = name == "emotion" ? ["dair-ai/emotion", "split"] : ["fancyzhx/ag_news", "default"]
      20.times do |batch|
        path = File.join(DOWNLOADS, "#{name}-#{format('%03d', batch)}.json")
        next if File.exist?(path)
        url = "https://datasets-server.huggingface.co/rows?dataset=#{dataset}&config=#{config}&split=test&offset=#{batch * 100}&length=100"
        metadata = EasyAI::Decision::Data::Download.fetch(url, path, max_mib: 10)
        puts JSON.generate(metadata)
      end
    end
  end

  def prepare(root)
    output = File.join(root, "generalization")
    raise ArgumentError, "Panel already exists" if File.exist?(output)
    FileUtils.mkdir_p(output)
    tokenizer = EasyAI::Tokenizers::Registry.load(File.join(root, "data/tokenizer.json"))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: EvidenceExperiment.config(1337, "answer"))
    used = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/mixed.jsonl")).map { |row| text_hash(row.state) }.to_set
    exclusions = Hash.new(0)
    rows = LABELS.flat_map do |source, labels|
      name = source == "Emotion" ? "emotion" : "ag_news"
      pool = 20.times.flat_map { |batch| JSON.parse(File.read(File.join(DOWNLOADS, "#{name}-#{format('%03d', batch)}.json"))).fetch("rows") }
      raise ArgumentError, "Dataset-server truncated cells" if pool.any? { |row| row.fetch("truncated_cells", []).any? }
      seen = Set.new
      pool.sort_by { |row| Digest::SHA256.hexdigest("generalization-v1:#{source}:#{row.fetch('row_idx')}") }.filter_map do |wrapped|
        raw = wrapped.fetch("row")
        hash = text_hash(raw.fetch("text"))
        if used.include?(hash) || !seen.add?(hash)
          exclusions["#{source}/duplicate_or_train_overlap"] += 1
          next
        end
        example = build(source, "en-US", hash, raw.fetch("text"),
          source == "Emotion" ? "Which emotion does the text express?" : "What is the topic of this news article?",
          labels.each_with_index.to_h { |text, i| [i.to_s, text] }, raw.fetch("label").to_s)
        next unless within_limits?(example, collator, exclusions)
        example
      end.first(1000)
    end
    locales = massive_locales
    shared = locales.values.map { |items| items.map { |row| row.fetch("id").to_s }.to_set }.reduce(:&)
    ids = shared.sort_by { |id| Digest::SHA256.hexdigest("generalization-v1:massive:#{id}") }.first(900).to_set
    locales.each do |locale, items|
      seen = Set.new
      items.select { |row| ids.include?(row.fetch("id").to_s) }.each do |raw|
        descriptions = SCENARIOS.transform_values { |texts| texts.fetch(locale == "zh-CN" ? 0 : 1) }
        hash = text_hash(raw.fetch("utt"))
        if used.include?(hash) || !seen.add?(hash)
          exclusions["MASSIVE/duplicate_or_train_overlap"] += 1
          next
        end
        example = build("MASSIVE-Scenario", locale, raw.fetch("id").to_s, raw.fetch("utt"),
          locale == "zh-CN" ? "这项请求属于哪个领域？" : "Which domain does this request belong to?", descriptions, raw.fetch("scenario"))
        rows << example if within_limits?(example, collator, exclusions)
      end
    end
    EvidenceExperiment.write_rows(File.join(output, "test.jsonl"), rows)
    files = Dir.glob(File.join(DOWNLOADS, "*.json")) + [File.join(ROOT, "data/decision/downloads/massive-1.1.tar.gz")]
    manifest = { "version" => 1, "created_at" => Time.now.utc.iso8601, "rows" => rows.size,
      "counts" => rows.group_by { |row| "#{row['source']}/#{row['language']}" }.transform_values(&:size),
      "exclusions" => exclusions, "test_sha256" => Digest::SHA256.file(File.join(output, "test.jsonl")).hexdigest,
      "sources_sha256" => files.to_h { |path| [File.expand_path(path), Digest::SHA256.file(path).hexdigest] },
      "sources" => { "AG-News" => "https://huggingface.co/datasets/fancyzhx/ag_news", "Emotion" => "https://huggingface.co/datasets/dair-ai/emotion", "MASSIVE" => EasyAI::Decision::Data::Adapters::Massive::URL },
      "scope" => "Supplemental zero-shot diagnostic added during training, before test evaluation. 1000 deterministic unique examples from each 2000-row public test pool for news/emotion; up to 900 shared MASSIVE official test IDs in each of zh-CN/en-US, 18 scenarios. Full label sets, no target-conditioned negative sampling. No training, checkpoint selection, new calibration fitting or new acceptance claim. Dataset-server snapshots are hash-recorded, not pinned upstream revisions; retain local raw files for exact reproduction. Cross-language MASSIVE rows share groups." }
    SemanticCoverage.write_json(File.join(output, "manifest.json"), manifest)
    puts JSON.pretty_generate(manifest.slice("rows", "counts", "exclusions"))
  end

  def report(root)
    directory = File.join(root, "generalization")
    results = (%w[parent] + EvidenceExperiment::ARMS).to_h do |arm|
      [arm, EvidenceExperiment::SEEDS.map { |seed| JSON.parse(File.read(File.join(directory, "#{arm}-#{seed}.json"))) }]
    end
    tasks = results.fetch("parent").first.fetch("tasks").keys
    means = results.transform_values do |runs|
      tasks.to_h do |task|
        rows = runs.map { |run| run.fetch("tasks").fetch(task) }
        [task, { "count" => rows.first["count"], "accuracy" => EvidenceEvaluation.average(rows.map { |row| row["accuracy"] }),
          "seed_accuracies" => rows.map { |row| row["accuracy"] },
          "balanced_accuracy" => EvidenceEvaluation.average(rows.map { |row| row["balanced_accuracy"] }),
          "nll" => EvidenceEvaluation.average(rows.map { |row| row["nll"] }),
          "majority_accuracy" => rows.first["majority_accuracy"], "chance_accuracy" => rows.first["chance_accuracy"] }]
      end
    end
    report = { "means" => means, "scope" => "Zero-shot transfer diagnostic, not an acceptance gate. Four task/language cells from three new task families. Scores are not directly comparable to other models' protocols." }
    SemanticCoverage.write_json(File.join(directory, "report.json"), report)
    table = "Task                          Parent   Answer   Evidence  Majority  Chance\n"
    tasks.each do |task|
      values = %w[parent answer evidence].map { |arm| means[arm][task]["accuracy"] * 100 }
      values += means["parent"][task].values_at("majority_accuracy", "chance_accuracy").map { |value| value * 100 }
      table << format("%-29s %7.2f %8.2f %9.2f %9.2f %7.2f\n", task, *values)
    end
    File.write(File.join(directory, "summary.txt"), table)
    puts table
    report
  end

  def text_hash(text)
    Digest::SHA256.hexdigest(text.unicode_normalize(:nfkc).strip.gsub(/\s+/, " "))
  end

  def build(source, language, origin, state, question, labels, target)
    group = "generalization:#{source}:#{origin}"
    options = labels.map { |id, text| { "id" => id, "text" => text } }.shuffle(random: Random.new(Digest::SHA256.hexdigest(group).to_i(16)))
    { "id" => "#{group}:#{language}", "group_id" => group, "source" => source, "language" => language,
      "state" => state, "question" => question, "options" => options, "target" => target }
  end

  def within_limits?(row, collator, exclusions)
    collator.state_tokens(row.fetch("state"))
    row.fetch("options").each { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }
    true
  rescue ArgumentError => error
    raise unless error.message.include?("exceeds")
    exclusions["#{row.fetch('source')}/over_length"] += 1
    false
  end

  def massive_locales
    rows = {}
    Zlib::GzipReader.open(File.join(ROOT, "data/decision/downloads/massive-1.1.tar.gz")) do |gzip|
      Gem::Package::TarReader.new(gzip) do |tar|
        tar.each do |entry|
          name = File.basename(entry.full_name, ".jsonl")
          next unless entry.file? && %w[en-US zh-CN].include?(name)
          rows[name] = entry.read.lines.map { |line| JSON.parse(line) }.select { |row| row.fetch("partition") == "test" }
        end
      end
    end
    raise ArgumentError, "Missing locales" unless rows.keys.sort == %w[en-US zh-CN]
    rows
  end

  def evaluate(root, arm, seed)
    EvidenceEvaluation.assert_complete(root)
    directory = File.join(root, "generalization")
    manifest = JSON.parse(File.read(File.join(directory, "manifest.json")))
    data = EasyAI::Decision::Data::Dataset.new(File.join(directory, "test.jsonl"))
    raise ArgumentError, "Panel changed" unless data.fingerprint == manifest.fetch("test_sha256")
    primary = JSON.parse(File.read(File.join(root, "evaluation-#{arm}-#{seed}.json")))
    loaded = EasyAI::Decision::Checkpoint.load(primary.fetch("checkpoint"))
    raise ArgumentError, "Weights changed" unless loaded[:weights_fingerprint] == primary.fetch("weights_sha256")
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    evaluator = SemanticCoverageEvaluation.new(model: loaded[:model], tokenizer: loaded[:tokenizer], device: device, batch_size: 4)
    calibrator = EasyAI::Decision::Calibrator.new(temperature: primary.fetch("temperature"))
    results = data.to_a.group_by { |row| "#{row.source}/#{row.language}" }.transform_values do |rows|
      logits = evaluator.collect(rows)
      metrics = EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index), calibrator)
      labels = rows.first.options.map { |option| option.fetch("id") }
      counts = rows.group_by(&:target).transform_values(&:size)
      metrics["chance_accuracy"] = 1.0 / labels.size
      metrics["majority_accuracy"] = counts.values.max.fdiv(rows.size)
      metrics["by_label"] = rows.each_index.group_by { |i| rows[i].target }.transform_values do |indexes|
        { "count" => indexes.size, "accuracy" => indexes.count { |i| logits[i].each_index.max_by { |j| logits[i][j] } == rows[i].target_index }.fdiv(indexes.size) }
      end
      metrics["balanced_accuracy"] = metrics["by_label"].values.sum { |row| row["accuracy"] }.fdiv(labels.size)
      metrics["independent_groups"] = rows.map(&:group_id).uniq.size
      metrics
    end
    SemanticCoverage.write_json(File.join(directory, "#{arm}-#{seed}.json"), { "arm" => arm, "seed" => seed, "tasks" => results,
      "macro_accuracy" => results.values.sum { |row| row["accuracy"] }.fdiv(results.size),
      "worst_task_accuracy" => results.values.map { |row| row["accuracy"] }.min,
      "calibration" => "Transferred primary calibration temperature, no panel-label fitting" })
    puts "Generalization #{arm}/#{seed}: #{results.transform_values { |row| row['accuracy'].round(4) }}"
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/evidence-v1", arm: "answer", seed: 1337 }
  OptionParser.new do |parser|
    %i[phase output arm].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "download" then DecisionGeneralization.download
  when "prepare" then DecisionGeneralization.prepare(root)
  when "evaluate" then DecisionGeneralization.evaluate(root, options[:arm], options[:seed])
  when "report" then DecisionGeneralization.report(root)
  else raise ArgumentError, "Expected download, prepare, evaluate or report"
  end
end
