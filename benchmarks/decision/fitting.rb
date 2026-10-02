#!/usr/bin/env ruby
require_relative "evidence_evaluation" unless defined?(EvidenceEvaluation)

# Diagnostic-only fitting: full candidates are permuted, never resampled/dropped.
class DecisionFittingTrainer < EasyAI::Decision::Trainer
  def self.permute(examples, seed)
    rng = Random.new(seed ^ 0xF177)
    examples.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.shuffle(random: rng))) }
  end

  private

  def loss_for(examples, seed:, teacher: false)
    super(self.class.permute(examples, seed), seed: seed, teacher: teacher)
  end
end

module DecisionFitting
  SEED = 1337
  STEPS = 2000
  STARTS = %w[parent scratch].freeze
  POSITIONS = %w[sinusoidal rotary].freeze
  module_function

  def config(position)
    raise ArgumentError, "Unknown position encoding" unless POSITIONS.include?(position)
    EvidenceExperiment.config(SEED, "answer").with(model: { position_encoding: position, dropout: 0.0, evidence_head: false },
      training: { steps: STEPS, balance_sources: false, balance_labels: false, paired_sampling: false,
        early_stopping_patience: 0, checkpoint_every: 500, track_coverage: true })
  end

  def prepare(root)
    raise ArgumentError, "Output already exists" if File.exist?(root)
    previous = File.join(EvidenceExperiment::ROOT, "runs/decision/evidence-v1")
    EvidenceExperiment.verify(previous)
    loaded = EasyAI::Decision::Checkpoint.load(EvidenceExperiment.parent(SEED))
    FileUtils.mkdir_p(root)
    FileUtils.cp(File.join(previous, "data/sanity.jsonl"), File.join(root, "train.jsonl"))
    tokenizer = loaded.fetch(:tokenizer)
    tokenizer.save(File.join(root, "tokenizer.json"))
    rows = EasyAI::Decision::Data::Dataset.new(File.join(root, "train.jsonl"))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config("sinusoidal"))
    rows.each_slice(16) { |batch| collator.call(batch) }
    tokenized = rows.map { |row| collator.state_tokens(row.state) }
    initial = STARTS.to_h do |start|
      Torch.manual_seed(SEED)
      model = EasyAI::Decision::ChoiceModel.new(config("sinusoidal"))
      model.load_state_dict(loaded.fetch(:model).state_dict) if start == "parent"
      path = EasyAI::Decision::Checkpoint.save(File.join(root, "initial-#{start}"), model: model, tokenizer: tokenizer)
      [start, { "checkpoint" => path, "weights_sha256" => Digest::SHA256.file(File.join(path, "weights.pt")).hexdigest }]
    end
    protocol = { "seed" => SEED, "steps" => STEPS, "rows" => rows.size, "unique_states" => rows.map(&:state).uniq.size,
      "unique_tokenized_states" => tokenized.uniq.size, "training_sha256" => Digest::SHA256.file(File.join(root, "train.jsonl")).hexdigest,
      "tokenizer_fingerprint" => tokenizer.fingerprint, "parent_weights_sha256" => loaded.fetch(:weights_fingerprint), "initial" => initial,
      "configs" => POSITIONS.to_h { |position| [position, config(position).to_h] },
      "scope" => "Single-seed 2x2 TRAINING-SET diagnostic, no generalization claim. Same 128 rows, tokenizer, tensors within each start, optimizer, schedule, full-candidate permutations and sampling. RoPE switches positional function, not tensor shapes. Parent transfer may suffer representation mismatch. Dropout/evidence loss off; optimizer resets for every fit. Fixed 2000 updates, no best-checkpoint selection. Gradient norms measured after clipping/update, not raw preclip norms. Perturbed states retain original labels and measure sensitivity only." }
    SemanticCoverage.write_json(File.join(root, "protocol.json"), protocol)
    puts "Prepared fitting diagnostic: #{rows.size} rows, #{tokenized.uniq.size} tokenized states"
  end

  def groups(raw, correct)
    raw.first.fetch("world").fetch("checks").keys.to_h do |name|
      sets = raw.each_index.group_by { |i| raw[i].fetch("world").fetch("checks").fetch(name) }.values
      [name, { "count" => sets.size, "all_correct" => sets.count { |indexes| indexes.all? { |i| correct[i] } }.fdiv(sets.size) }]
    end
  end

  def gradients(model)
    values = model.named_parameters.group_by { |name, _| name.split('.').first }.transform_values do |parameters|
      parameters.sum { |_name, tensor| tensor.grad ? tensor.grad.detach.pow(2).sum.item : 0.0 }**0.5
    end
    raise FloatDomainError, "Non-finite gradient norm" unless values.values.all?(&:finite?)
    values
  end

  def measure(trainer, rows, raw, step)
    evaluator = SemanticCoverageEvaluation.new(model: trainer.model, tokenizer: trainer.tokenizer, device: trainer.device, batch_size: 16)
    logits = evaluator.collect(rows)
    correct = rows.each_index.map { |i| logits[i].each_index.max_by { |j| logits[i][j] } == rows[i].target_index }
    metrics = EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index))
    metrics.merge("step" => step, "device" => trainer.device, "groups" => groups(raw, correct), "gradients_postclip" => gradients(trainer.model),
      "logit_margin_std" => deviation(logits.map { |scores| scores[0] - scores[1] }))
  end

  def deviation(values)
    mean = values.sum.fdiv(values.size)
    Math.sqrt(values.sum { |value| (value - mean)**2 }.fdiv(values.size))
  end

  def final_diagnostics(trainer, rows, raw)
    evaluator = SemanticCoverageEvaluation.new(model: trainer.model, tokenizer: trainer.tokenizer, device: trainer.device, batch_size: 16)
    original = evaluator.collect(rows)
    reversed = evaluator.collect(rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) })
    changed = rows.each_index.group_by { |i| rows[i].language }.values.flat_map do |indexes|
      states = indexes.map { |i| rows[i].state }.shuffle(random: Random.new(SEED))
      indexes.each_with_index.map { |i, j| [i, EasyAI::Decision::Data::Example.new(rows[i].to_h.merge("state" => states[j]))] }
    end.sort_by(&:first).map(&:last)
    other = evaluator.collect(changed)
    failures = rows.each_index.filter_map do |i|
      next if original[i].each_index.max_by { |j| original[i][j] } == rows[i].target_index
      raw[i].merge("logits" => original[i])
    end
    { "failures" => failures, "shuffled_state" => EasyAI::Decision::Evaluator.metrics(other, rows.map(&:target_index)),
      "permutation_max_logit_error" => original.each_index.flat_map { |i| original[i].zip(reversed[i].reverse).map { |a, b| (a - b).abs } }.max,
      "shuffled_max_logit_change" => original.each_index.flat_map { |i| original[i].zip(other[i]).map { |a, b| (a - b).abs } }.max,
      "perturbation_scope" => "Shuffled within language; original labels retained, not new gold. Sensitivity is not proof of reasoning." }
  end

  def train(root, start, position)
    raise ArgumentError, "Unknown start" unless STARTS.include?(start)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    path = File.join(root, "train.jsonl")
    raise ArgumentError, "Training set changed" unless Digest::SHA256.file(path).hexdigest == protocol.fetch("training_sha256")
    initial = protocol.fetch("initial").fetch(start)
    loaded = EasyAI::Decision::Checkpoint.load(initial.fetch("checkpoint"))
    raise ArgumentError, "Initial tensors changed" unless loaded.fetch(:weights_fingerprint) == initial.fetch("weights_sha256")
    raise ArgumentError, "Tokenizer changed" unless loaded.fetch(:tokenizer).fingerprint == protocol.fetch("tokenizer_fingerprint")
    cfg = EasyAI::Decision::Config.new(protocol.fetch("configs").fetch(position))
    model = EasyAI::Decision::ChoiceModel.new(cfg)
    model.load_state_dict(loaded.fetch(:model).state_dict)
    delta = model.state_dict.map { |name, value| (value - loaded.fetch(:model).state_dict.fetch(name)).abs.max.item }.max
    raise "Initial weight mismatch" unless delta.zero?
    loaded[:model] = nil
    GC.start
    directory = File.join(root, "#{start}-#{position}")
    raise ArgumentError, "Run already exists" if File.exist?(directory)
    FileUtils.mkdir_p(directory)
    ENV["EASY_AI_LOG_PATH"] = File.join(directory, "train.log")
    EasyAI::Logger.reset!
    dataset = EasyAI::Decision::Data::Dataset.new(path)
    rows, raw = dataset.to_a, EvidenceEvaluation.raw_rows(path)
    trainer = DecisionFittingTrainer.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: File.join(directory, "choice"))
    measurements = [measure(trainer, rows, raw, 0)]
    trace = File.join(directory, "diagnostics.jsonl")
    File.write(trace, JSON.generate(measurements.first) + "\n")
    progress = EasyAI::Decision::Progress.new(task: "#{start}-#{position}", total: STEPS)
    trainer.train do |state, loss|
      progress.update(state, loss, device: trainer.device)
      next unless (state["step"] % 100).zero?
      measurement = measure(trainer, rows, raw, state["step"])
      measurements << measurement
      File.open(trace, "a") { |file| file.puts(JSON.generate(measurement)) }
      puts "Fit #{start}/#{position} step=#{state['step']} accuracy=#{measurement['accuracy'].round(4)} fact_flip=#{measurement.dig('groups', 'fact_flip', 'all_correct').round(4)}"
    end
    final = final_diagnostics(trainer, rows, raw)
    EvidenceExperiment.write_rows(File.join(directory, "failures.jsonl"), final.delete("failures"))
    summary = { "start" => start, "position" => position, "step" => trainer.state["step"], "checkpoint" => trainer.last_checkpoint,
      "initial_tensor_max_difference" => delta, "final" => measurements.last, "diagnostics" => final,
      "row_visits" => trainer.state.fetch("coverage").fetch("row_visits"), "parameters" => model.parameter_count,
      "fit_gate" => measurements.last.fetch("accuracy") >= 0.99 && measurements.last.dig("groups", "fact_flip", "all_correct") >= 0.95 }
    SemanticCoverage.write_json(File.join(directory, "summary.json"), summary)
  end

  def report(root)
    rows = STARTS.product(POSITIONS).map { |start, position| JSON.parse(File.read(File.join(root, "#{start}-#{position}/summary.json"))) }
    raise "Incomplete diagnostic budget" unless rows.all? { |row| row.fetch("step") == STEPS }
    raise "Different sampling schedules" unless rows.map { |row| row.fetch("row_visits") }.uniq.size == 1
    examples = EvidenceEvaluation.raw_rows(File.join(root, "train.jsonl"))
    totals = examples.group_by { |row| row.fetch("language") }.transform_values(&:size)
    rows.each do |row|
      directory = File.join(root, "#{row.fetch('start')}-#{row.fetch('position')}")
      trace = EvidenceEvaluation.raw_rows(File.join(directory, "diagnostics.jsonl"))
      first = trace.find { |item| item.fetch("accuracy") >= 0.99 && item.dig("groups", "fact_flip", "all_correct") >= 0.95 }
      row["first_evaluated_fit_gate_step"] = first && first.fetch("step")
      failures = EvidenceEvaluation.raw_rows(File.join(directory, "failures.jsonl")).group_by { |item| item.fetch("language") }
      row["training_accuracy_by_language"] = totals.to_h { |language, total| [language, 1.0 - failures.fetch(language, []).size.fdiv(total)] }
    end
    report = { "scope" => "Single-seed TRAINING-SET diagnostic, not generalization or production acceptance", "runs" => rows }
    SemanticCoverage.write_json(File.join(root, "report.json"), report)
    plot(root)
    rows.each { |row| puts "#{row['start']}/#{row['position']}: accuracy=#{row.dig('final', 'accuracy')} fit_gate=#{row['fit_gate']}" }
    report
  end

  def plot(root)
    lines = STARTS.product(POSITIONS).map do |start, position|
      name = "#{start}-#{position}"
      rows = EvidenceEvaluation.raw_rows(File.join(root, name, "diagnostics.jsonl"))
      File.write(File.join(root, "#{name}.tsv"), rows.map { |row| [row["step"], row["nll"], row["accuracy"], row.dig("groups", "fact_flip", "all_correct"), row.dig("groups", "binding", "all_correct")].join("\t") }.join("\n") + "\n")
      name
    end
    script = "set terminal pngcairo size 1200,850\nset output 'comparison.png'\nset multiplot layout 2,2 title 'Fitting diagnostic: training set only, seed 1337'\nset xlabel 'Update'\nset key bottom right\n"
    [[2, "Answer NLL"], [3, "Accuracy"], [4, "Fact-flip all correct"], [5, "Binding all correct"]].each do |column, title|
      script << "set title '#{title}'\nplot #{lines.map { |name| "'#{name}.tsv' using 1:#{column} with linespoints title '#{name}'" }.join(', ')}\n"
    end
    script << "unset multiplot\n"
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Plot failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/fitting-v1", start: "parent", position: "sinusoidal" }
  OptionParser.new do |parser|
    %i[phase output start position].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "prepare" then DecisionFitting.prepare(root)
  when "train" then DecisionFitting.train(root, options[:start], options[:position])
  when "report" then DecisionFitting.report(root)
  when "all"
    DecisionFitting.prepare(root)
    DecisionFitting::STARTS.product(DecisionFitting::POSITIONS).each do |start, position|
      raise "Fitting run failed" unless system(RbConfig.ruby, __FILE__, "--phase", "train", "--output", root, "--start", start, "--position", position)
    end
    DecisionFitting.report(root)
  else raise ArgumentError, "Unknown phase"
  end
end
