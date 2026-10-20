require "tmpdir"

module DecisionScaling
  module_function

  def batch_token_slots(lengths)
    states = lengths.map(&:first)
    options = lengths.map(&:last)
    raise ArgumentError, "Padding audit needs uniform candidate counts" unless !lengths.empty? && options.map(&:size).uniq.size == 1
    { "unpadded" => states.sum + options.flatten.sum,
      "padded" => states.size * states.max + options.size * options.first.size * options.flatten.max }
  end

  def padding_audit(root)
    protocol = verify(root)
    tokenizer = EasyAI::Decision::Checkpoint.load(PARENT).fetch(:tokenizer)
    replayed = {}
    values = Dir[File.join(root, "arms/*/{screen,extended}.json")].sort.to_h do |path|
      cell = File.basename(File.dirname(path))
      profile, scale, seed = cell.split("-")
      result = JSON.parse(File.read(path))
      config = EasyAI::Decision::Config.new(protocol.fetch("config_by_profile").fetch(profile))
      key = [scale, seed, result.fetch("step"), config[:training]["choice_microbatch"]]
      counts = replayed[key] ||= Dir.mktmpdir("decision-padding-") do |directory|
        dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data", "train-#{scale}-binary.jsonl"))
        # The model is never called: use a tiny CPU placeholder to construct the
        # exact production sampler without allocating a fitted model or GPU.
        tiny = config.with(model: { hidden_size: 16, encoder_layers: 1, attention_heads: 2, ffn_size: 32, ffn_sizes: [32] }, training: { device: "cpu" })
        sampler = QuadTrainer.new(model: EasyAI::Decision::ChoiceModel.new(tiny), tokenizer: tokenizer, dataset: dataset, output: directory)
        collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config)
        states, options = {}, {}
        lengths = dataset.map do |row|
          state = states[row.state] ||= collator.state_tokens(row.state).size
          candidates = row.options.map { |option| options[[row.question, option.fetch("text")]] ||= collator.option_tokens(row.question, option.fetch("text")).size }
          [state, candidates]
        end
        visits = Array.new(dataset.size, 0)
        totals = { "unpadded" => 0, "padded" => 0 }
        result.fetch("step").times do |step|
          indices = sampler.send(:sample_indices, 32, rng: Random.new(Integer(seed) + step * 1009))
          indices.each { |index| visits[index] += 1 }
          indices.each_slice(config[:training]["choice_microbatch"]) do |micro|
            batch_token_slots(micro.map { |index| lengths.fetch(index) }).each { |name, count| totals[name] += count }
          end
        end
        totals.merge("row_visits" => visits)
      end
      coverage = JSON.parse(File.read(File.join(result.fetch("checkpoint"), "metadata.json"))).fetch("training").fetch("coverage")
      raise "Sampler/token replay mismatch" unless counts.fetch("row_visits") == coverage.fetch("row_visits") && counts.fetch("unpadded") == coverage.fetch("input_tokens")
      summary = counts.except("row_visits").merge("padding_fraction" => 1.0 - counts.fetch("unpadded").fdiv(counts.fetch("padded")),
        "exact_row_visit_and_token_replay" => true, "result_sha256" => sha(path))
      puts "Padding #{cell}/#{File.basename(path, '.json')}: #{summary.fetch('padding_fraction').round(4)}"
      $stdout.flush
      ["#{cell}/#{File.basename(path, '.json')}", summary]
    end
    write(File.join(root, "padding-audit.json"), { "endpoints" => values,
      "scope" => "Offline CPU replay of the exact controlled sampler; metadata row visits and unpadded tokens must match. Padded state/option encoder input slots, not FLOPs or matching-kernel profiling. No forward/backward or optimizer update; tiny constructor checkpoints are temporary and discarded." })
  end
end
