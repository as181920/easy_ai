module DecisionScaling
  module_function

  def model_config(parent, profile, microbatch: 4, steps: 4000, seed: 1337)
    layers, width = PROFILES.fetch(profile.to_s) { raise ArgumentError, "Unknown capacity profile" }
    raise ArgumentError, "Microbatch must divide 32" unless [1, 2, 4, 8, 16, 32].include?(microbatch)
    parent.config.with(model: { encoder_layers: layers, ffn_size: width, ffn_sizes: Array.new(layers, width),
      encoding_mode: :separate, position_encoding: :rotary, pooling: :all, dropout: 0.0 },
      training: { seed: seed, steps: steps, learning_rate: 0.0003, warmup_steps: 0, choice_microbatch: microbatch,
        gradient_accumulation: 32 / microbatch, checkpoint_every: 250, eval_every: 250, early_stopping_patience: 0,
        paired_sampling: false, balance_sources: false, balance_labels: false, resample_negatives: false, track_coverage: true },
      runtime: { gpu_memory_budget_mib: 4096 }, growth: { enabled: false })
  end

  def initialize_model(profile, microbatch: 4, steps: 4000, seed: 1337)
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    target = model_config(loaded.fetch(:model), profile, microbatch: microbatch, steps: steps, seed: seed)
    base = target.with(model: loaded.fetch(:model).config[:model].merge("position_encoding" => "rotary", "dropout" => 0.0))
    original, transfer = EasyAI::Decision::WeightTransfer.apply(parent: loaded, config: base, tokenizer: loaded.fetch(:tokenizer), seed: seed)
    model = EasyAI::Decision::Growth::Scale.apply(original, layers: target[:model]["encoder_layers"], ffn_size: target[:model]["ffn_size"], seed: seed)
    raise "Capacity config mismatch" unless model.config.to_h == target.to_h
    [model, loaded.fetch(:tokenizer), transfer.merge("profile" => profile.to_s, "scaled_parameters" => model.parameter_count,
      "initialization" => "Own v0.1 transfer plus function-preserving depth/FFN expansion; fresh optimizer")]
  end
end
