#!/usr/bin/env ruby
require_relative "fitting" unless defined?(DecisionFitting)
require_relative "generalization" unless defined?(DecisionGeneralization)

# Same mean per-example CE, without padding smaller candidate sets to eighteen.
class NaturalTrainer < DecisionFittingTrainer
  private

  def loss_for(examples, seed:, teacher: false)
    @grouped_input_tokens = 0
    components = Hash.new(0.0)
    losses = examples.group_by { |row| row.options.size }.values.map do |rows|
      weight = rows.size.fdiv(examples.size)
      loss = super(rows, seed: seed, teacher: teacher) * weight
      @grouped_input_tokens += @collator.input_token_counts.sum
      @batch_losses.each { |key, value| components[key] += value * weight } if teacher
      loss
    end
    @batch_losses = components if teacher
    Torch.stack(losses).sum
  end

  def input_token_count
    @grouped_input_tokens
  end
end

module DecisionNatural
  SEED = 1337
  STEPS = 1000
  MIXTURES = %w[control broad].freeze
  POSITIONS = %w[sinusoidal rotary].freeze
  DOWNLOADS = File.join(EvidenceExperiment::ROOT, "data/decision/downloads/natural-v1")
  NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv/train.csv".freeze
  NEWS_SHA256 = "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e".freeze
  module_function

  def download
    path = File.join(DOWNLOADS, "ag-news-train.csv")
    if File.file?(path)
      raise "News cache changed" unless Digest::SHA256.file(path).hexdigest == NEWS_SHA256
      return
    end
    provenance = EasyAI::Decision::Data::Download.fetch(NEWS_URL, path, expected_sha256: NEWS_SHA256)
    SemanticCoverage.write_json("#{path}.source.json", provenance)
  end

  def all(root)
    raise ArgumentError, "Output exists" if File.exist?(root)
    download
    raise "GPU profile failed" unless system(RbConfig.ruby, File.join(__dir__, "natural_memory_profile.rb"), "4")
    prepare(root)
    MIXTURES.product(POSITIONS).each do |mixture, position|
      raise "Training failed" unless system(RbConfig.ruby, __FILE__, "--phase", "train", "--output", root, "--mixture", mixture, "--position", position)
    end
    raise "Evaluation failed" unless system(RbConfig.ruby, File.join(__dir__, "natural_evaluation.rb"), "--phase", "all", "--output", root)
  end

  def config(position)
    raise ArgumentError, "Unknown position" unless POSITIONS.include?(position)
    EvidenceExperiment.config(SEED, "answer").with(model: { position_encoding: position, evidence_head: false, dropout: 0.1 },
      training: { steps: STEPS, warmup_steps: 100, eval_every: 200, checkpoint_every: 500,
        choice_microbatch: 4, gradient_accumulation: 8, balance_sources: true, track_coverage: true })
  end

  def known_material
    files = Dir.glob(File.join(EvidenceExperiment::ROOT, "data/decision/**/{test,validation,calibration}.jsonl")) +
      %w[semantic-coverage-v2/data/challenge.jsonl evidence-v1/data/public-test.jsonl evidence-v1/data/test.jsonl
        evidence-v1/generalization/test.jsonl evidence-v1/shared-benchmarks/test.jsonl].map { |name| File.join(EvidenceExperiment::ROOT, "runs/decision", name) }
    files << File.join(EvidenceExperiment::ROOT, "data/decision/semantic-public/train.jsonl")
    files = files.select { |path| File.file?(path) }.uniq
    hashes = files.each_with_object(Set.new) do |path, set|
      File.foreach(path) do |line|
        row = JSON.parse(line)
        set << EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state")) if row["state"]
      end
    end
    [hashes, files]
  end

  def subset(rows, per_language:, salt:)
    rows.group_by { |row| row.fetch("language") }.values.flat_map do |items|
      items.sort_by { |row| Digest::SHA256.hexdigest("#{salt}:#{row.fetch('id')}") }.first(per_language)
    end
  end

  def common_panel(old_path, added, salt:, blocked:)
    original = EvidenceEvaluation.raw_rows(old_path).reject { |row| blocked.include?(EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state"))) }.group_by { |row| row.fetch("source") }.values.flat_map do |rows|
      selected = subset(rows, per_language: 96, salt: salt)
      raise "Insufficient old validation/calibration source" unless selected.size == 96
      selected
    end
    news = subset(added.select { |row| row.fetch("source") == "AG-News" }, per_language: 96, salt: salt)
    massive = subset(added.select { |row| row.fetch("source") == "MASSIVE-Scenario" }, per_language: 48, salt: salt)
    raise "Insufficient new validation/calibration sources" unless news.size == 96 && massive.size == 96
    original + news + massive
  end

  def fresh_public(blocked, collator)
    raw = EasyAI::Decision::Data::SemanticAdapter.each(File.join(EvidenceExperiment::ROOT, "data/decision/downloads/semantics")).to_a
    groups = EasyAI::Decision::Data::SemanticCorpus.new(raw).split_rows.select { |_row, _group, split| split == "test" }
    exclusions = Hash.new(0)
    corpus = groups.filter_map do |row, group, _split|
      if blocked.include?(EasyAI::Decision::Data::NaturalCorpus.material(row.fetch(:state)))
        exclusions["#{row[:source]}/historical_overlap"] += 1
        next
      end
      result = EasyAI::Decision::Data::NaturalAdapter.build(row.fetch(:source), row.fetch(:language), group, row.fetch(:state),
        row.fetch(:question), row.fetch(:options), row.fetch(:target), "heldout")
      result["id"] = Digest::SHA256.hexdigest([group, row.fetch(:question)].join("\n"))
      result["group_id"] = group
      DecisionGeneralization.within_limits?(result, collator, exclusions) ? result : nil
    end
    selected = corpus.group_by { |row| row.fetch("source") }.values.flat_map do |rows|
      ids = rows.map { |row| row.fetch("group_id") }.uniq.sort_by { |id| Digest::SHA256.hexdigest("natural-public-v1:#{id}") }.first(150).to_set
      rows.select { |row| ids.include?(row.fetch("group_id")) }.uniq { |row| row.fetch("id") }
    end
    [selected, exclusions]
  end

  def semantic_sources
    manifest = JSON.parse(File.read(File.join(EvidenceExperiment::ROOT, "runs/decision/semantic-coverage-v2/protocol.json"))).fetch("source_sha256")
    paths = Dir.glob(File.join(EvidenceExperiment::ROOT, "data/decision/downloads/semantics/*")).select { |path| File.file?(path) }
    raise "Semantic source cache is incomplete" unless paths.map { |path| File.basename(path) }.sort == manifest.keys.sort
    paths.each { |path| raise "Semantic source cache changed" unless Digest::SHA256.file(path).hexdigest == manifest.fetch(File.basename(path)) }
    paths
  end

  def prepare(root)
    raise ArgumentError, "Output exists" if File.exist?(root)
    profile = JSON.parse(File.read(File.join(EvidenceExperiment::ROOT, "tmp/natural-memory-profile-4.json")))
    raise "GPU profile exceeded budget" unless profile.fetch("samples").all? { |row| row.fetch("mib") <= 4096 }
    original = File.join(EvidenceExperiment::ROOT, "data/decision/semantic-public")
    loaded = EasyAI::Decision::Checkpoint.load(EvidenceExperiment.parent(SEED))
    tokenizer = loaded.fetch(:tokenizer)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config("rotary"))
    news_path = File.join(DOWNLOADS, "ag-news-train.csv")
    provenance = JSON.parse(File.read("#{news_path}.source.json"))
    raise "News cache changed" unless Digest::SHA256.file(news_path).hexdigest == NEWS_SHA256 && provenance.fetch("sha256") == NEWS_SHA256
    raise "News source changed" unless provenance.fetch("source") == NEWS_URL
    semantic_files = semantic_sources
    news = EasyAI::Decision::Data::NaturalAdapter.news(news_path).to_a
    raise "Wrong official news count" unless news.size == 120_000
    massive_path = File.join(EvidenceExperiment::ROOT, "data/decision/downloads/massive-1.1.tar.gz")
    natural = news + EasyAI::Decision::Data::NaturalAdapter.massive(massive_path).to_a
    blocked, exclusion_files = known_material
    corpus = EasyAI::Decision::Data::NaturalCorpus.new(natural, blocked: blocked, collator: collator)
    splits = corpus.splits
    public_test, public_exclusions = fresh_public(blocked, collator)
    emotion_files = Dir.glob(File.join(DecisionGeneralization::DOWNLOADS, "emotion-???.json")).sort
    old_manifest = JSON.parse(File.read(File.join(EvidenceExperiment::ROOT, "runs/decision/evidence-v1/generalization/manifest.json")))
    emotion_files.each do |path|
      raise "Emotion snapshot changed" unless Digest::SHA256.file(path).hexdigest == old_manifest.fetch("sources_sha256").fetch(File.expand_path(path))
    end
    emotion = emotion_files.flat_map { |path| EasyAI::Decision::Data::NaturalAdapter.emotion(path).to_a }
    emotion.reject! { |row| blocked.include?(EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state"))) }
    emotion.uniq! { |row| EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state")) }
    emotion.select! { |row| DecisionGeneralization.within_limits?(row, collator, public_exclusions) }
    emotion = subset(emotion, per_language: 400, salt: "natural-heldout-emotion-v1")
    training_material = EvidenceEvaluation.raw_rows(File.join(original, "train.jsonl")).map do |row|
      EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state"))
    end.to_set
    validation = common_panel(File.join(original, "validation.jsonl"), splits.fetch("validation"), salt: "natural-validation-v1", blocked: training_material)
    calibration_blocked = training_material | validation.map { |row| EasyAI::Decision::Data::NaturalCorpus.material(row.fetch("state")) }.to_set
    calibration = common_panel(File.join(original, "calibration.jsonl"), splits.fetch("calibration"), salt: "natural-calibration-v1", blocked: calibration_blocked)
    natural_test = splits.fetch("test").group_by { |row| row.fetch("source") }.values.flat_map { |rows| subset(rows, per_language: 256, salt: "natural-test-v1") }
    raise "Insufficient fresh evaluation" unless public_test.map { |row| row.fetch("source") }.uniq.size == 3 && natural_test.size == 768 && emotion.size == 400
    FileUtils.mkdir_p(File.join(root, "data"))
    tokenizer.save(File.join(root, "data/tokenizer.json"))
    FileUtils.cp(File.join(original, "train.jsonl"), File.join(root, "data/control.jsonl"))
    File.open(File.join(root, "data/broad.jsonl"), "w") do |file|
      File.foreach(File.join(root, "data/control.jsonl")) { |line| file.write(line) }
      splits.fetch("train").each { |row| file.puts(JSON.generate(row)) }
    end
    { "validation" => validation, "calibration" => calibration, "test" => public_test + natural_test, "heldout" => emotion }.each do |name, rows|
      EvidenceExperiment.write_rows(File.join(root, "data/#{name}.jsonl"), rows)
    end
    datasets = %w[broad validation calibration test heldout].map { |name| EasyAI::Decision::Data::Dataset.new(File.join(root, "data/#{name}.jsonl")) }
    group_sets = datasets.map(&:groups)
    group_sets.combination(2).each { |left, right| raise "Group split leakage" if (left & right).any? }
    states = datasets.map { |data| data.map { |row| EasyAI::Decision::Data::NaturalCorpus.material(row.state) }.to_set }
    states.combination(2).each { |left, right| raise "Material split leakage" if (left & right).any? }
    initial = EasyAI::Decision::Checkpoint.save(File.join(root, "initial"), model: loaded.fetch(:model), tokenizer: tokenizer)
    protocol = { "version" => 1, "seed" => SEED, "steps" => STEPS, "arms" => MIXTURES.product(POSITIONS),
      "configs" => POSITIONS.to_h { |position| [position, config(position).to_h] }, "initial" => initial,
      "initial_sha256" => Digest::SHA256.file(File.join(initial, "weights.pt")).hexdigest, "profile" => profile,
      "files_sha256" => Dir.glob(File.join(root, "data/*")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
      "counts" => datasets.to_h { |data| [File.basename(data.path), data.group_by { |row| "#{row.source}/#{row.language}" }.transform_values(&:size)] },
      "exclusions" => corpus.exclusions, "public_exclusions" => public_exclusions,
      "sources_sha256" => ([news_path, massive_path] + emotion_files + exclusion_files + semantic_files).uniq.to_h { |path| [path, Digest::SHA256.file(path).hexdigest] },
      "news_provenance" => provenance, "scope" => "Single-seed fixed-1000-update 2x2 generalization PILOT, not the proposed 3-seed/4000-step acceptance round. Old QA/NLI versus broader mixture adding full-candidate AG News and MASSIVE en/zh. Emotion is entirely withheld from training/selection/calibration; unused official test snapshot rows, never previously evaluated. Same own parent tensors, common equal-source validation/calibration (96 rows per source; MASSIVE 48 per language), source-balanced mean CE, full-candidate permutations, candidate-count grouping, dropout .1, micro4/acc8, no auxiliary loss. Equal additional updates/examples, NOT equal tokens/FLOPs or task-specific exposure. Main tests limited to supported lengths; excluded long inputs/duplicates recorded. No test selection, no new pretrained weights, no promoted predictor. Raw source licenses/provenance remain dataset-specific." }
    SemanticCoverage.write_json(File.join(root, "protocol.json"), protocol)
    puts JSON.pretty_generate(protocol.slice("counts", "exclusions", "public_exclusions"))
  end

  def verify(root)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    protocol.fetch("files_sha256").each do |name, digest|
      raise "Prepared data changed" unless Digest::SHA256.file(File.join(root, "data", name)).hexdigest == digest
    end
    protocol
  end

  def train(root, mixture, position)
    raise ArgumentError, "Unknown mixture" unless MIXTURES.include?(mixture)
    protocol = verify(root)
    loaded = EasyAI::Decision::Checkpoint.load(protocol.fetch("initial"))
    raise "Initial checkpoint changed" unless loaded.fetch(:weights_fingerprint) == protocol.fetch("initial_sha256")
    model = EasyAI::Decision::ChoiceModel.new(EasyAI::Decision::Config.new(protocol.fetch("configs").fetch(position)))
    model.load_state_dict(loaded.fetch(:model).state_dict)
    delta = model.state_dict.map { |name, tensor| (tensor - loaded.fetch(:model).state_dict.fetch(name)).abs.max.item }.max
    raise "Initial tensor mismatch" unless delta.zero?
    loaded[:model] = nil
    GC.start
    directory = File.join(root, "#{mixture}-#{position}")
    raise "Run exists" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    trainer = NaturalTrainer.new(model: model, tokenizer: loaded.fetch(:tokenizer), output: File.join(directory, "choice"),
      dataset: EasyAI::Decision::Data::Dataset.new(File.join(root, "data/#{mixture}.jsonl")),
      validation: EasyAI::Decision::Data::Dataset.new(File.join(root, "data/validation.jsonl")))
    progress = EasyAI::Decision::Progress.new(task: "#{mixture}-#{position}", total: STEPS)
    trainer.train { |state, loss| progress.update(state, loss, device: trainer.device) }
    selected = EasyAI::Decision::Checkpoint.resolve(File.join(directory, "choice/best"))
    FileUtils.cp_r(selected, File.join(directory, "selected"))
    coverage = trainer.state.fetch("coverage")
    SemanticCoverage.write_json(File.join(directory, "coverage.json"), coverage)
    SemanticCoverage.write_json(File.join(directory, "summary.json"), { "mixture" => mixture, "position" => position, "step" => trainer.state.fetch("step"),
      "selected_step" => trainer.state.fetch("best_step"), "validation_nll" => trainer.state.fetch("best_validation_loss"),
      "initial_tensor_difference" => delta, "device" => trainer.device, "input_tokens" => coverage.fetch("input_tokens"),
      "unique_rows_seen" => coverage.fetch("row_visits").count(&:positive?), "examples_seen" => trainer.state.fetch("examples_seen") })
    EasyAI::Decision::TrainingReport.new(directory).write
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/natural-v1-pilot", mixture: "control", position: "sinusoidal" }
  OptionParser.new do |parser|
    %i[phase output mixture position].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "download" then DecisionNatural.download
  when "all" then DecisionNatural.all(root)
  when "prepare" then DecisionNatural.prepare(root)
  when "train" then DecisionNatural.train(root, options[:mixture], options[:position])
  else raise ArgumentError, "Expected download, prepare, train or all"
  end
end
