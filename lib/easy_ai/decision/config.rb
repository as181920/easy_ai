require "yaml"

module EasyAI
  module Decision
    class Config
      DEFAULTS = {
        "model" => {
          "vocab_size" => 32_000,
          "hidden_size" => 256,
          "encoder_layers" => 4,
          "attention_heads" => 4,
          "ffn_size" => 768,
          "ffn_sizes" => nil,
          "interaction_layers" => 1,
          "position_scale" => 1.0,
          "position_encoding" => "sinusoidal",
          "embedding_norm" => false,
          "score_mode" => "linear",
          "pooling" => "all",
          "encoding_mode" => "separate",
          "dropout" => 0.1
        },
        "input" => {
          "state_max_tokens" => 256,
          "question_option_max_tokens" => 64,
          "truncation" => "error"
        },
        "training" => {
          "device" => "auto",
          "precision" => "float32",
          "choice_microbatch" => 4,
          "mlm_microbatch" => 8,
          "gradient_accumulation" => 8,
          "learning_rate" => 0.0003,
          "weight_decay" => 0.01,
          "grad_clip" => 1.0,
          "seed" => 1337,
          "warmup_steps" => 100,
          "steps" => 1000,
          "eval_every" => 100,
          "checkpoint_every" => 100,
          "early_stopping_patience" => 0,
          "selection_min_delta" => 0.001,
          "balance_labels" => false,
          "balance_sources" => false,
          "paired_sampling" => false,
          "resample_negatives" => false,
          "mask_probability" => 0.15
        },
        "runtime" => {
          "gpu_memory_budget_mib" => 4096,
          "candidate_chunk_size" => 16,
          "cache_entries" => 16
        },
        "growth" => {
          "enabled" => false,
          "patience" => 3,
          "min_delta" => 0.001,
          "trial_evaluations" => 3,
          "max_layers" => 6,
          "max_trials" => 2,
          "operation" => "add_block",
          "ffn_increment" => 256,
          "max_ffn_size" => 1536
        }
      }.freeze

      attr_reader :options

      def initialize(overrides = {})
        @options = deep_merge(DEFAULTS, stringify(overrides))
        validate!
      end

      def self.load(path)
        new(YAML.safe_load_file(path, aliases: false) || {})
      end

      def [](key)
        options.fetch(key.to_s)
      end

      def with(overrides)
        self.class.new(deep_merge(options, stringify(overrides)))
      end

      def to_h
        Marshal.load(Marshal.dump(options))
      end

      private

      def stringify(value)
        case value
        when Hash then value.to_h { |k, v| [k.to_s, stringify(v)] }
        when Array then value.map { |v| stringify(v) }
        else value
        end
      end

      def deep_merge(base, extra)
        raise ArgumentError, "Configuration sections must be mappings" unless extra.is_a?(Hash)
        unknown = extra.keys - base.keys
        raise ArgumentError, "Unknown configuration keys: #{unknown.join(', ')}" unless unknown.empty?
        base.to_h do |key, value|
          replacement = extra.fetch(key, value)
          [key, value.is_a?(Hash) ? deep_merge(value, replacement) : replacement]
        end
      end

      def validate!
        model = self[:model]
        %w[vocab_size hidden_size encoder_layers attention_heads ffn_size interaction_layers].each do |key|
          positive_integer!(model.fetch(key), key)
        end
        raise ArgumentError, "vocab_size must include 262 byte/special tokens" if model["vocab_size"] < 262
        raise ArgumentError, "hidden_size must be even and divisible by attention_heads" unless
          model["hidden_size"].even? && (model["hidden_size"] % model["attention_heads"]).zero?
        sizes = model["ffn_sizes"]
        if sizes
          raise ArgumentError, "ffn_sizes must match encoder_layers" unless sizes.is_a?(Array) && sizes.length == model["encoder_layers"]
          sizes.each { |n| positive_integer!(n, "ffn_sizes") }
        end
        raise ArgumentError, "dropout must be in [0,1)" unless (0...1).cover?(model["dropout"])
        position_scale = model["position_scale"]
        raise ArgumentError, "position_encoding must be sinusoidal or rotary" unless %w[sinusoidal rotary].include?(model["position_encoding"])
        if model["position_encoding"] == "rotary" && (model["hidden_size"] / model["attention_heads"]).odd?
          raise ArgumentError, "Rotary attention needs even head width"
        end
        raise ArgumentError, "embedding_norm must be boolean" unless [true, false].include?(model["embedding_norm"])
        raise ArgumentError, "score_mode must be linear or matching" unless %w[linear matching].include?(model["score_mode"])
        raise ArgumentError, "pooling must be all or candidate" unless %w[all candidate].include?(model["pooling"])
        raise ArgumentError, "encoding_mode must be separate or joint" unless %w[separate joint].include?(model["encoding_mode"])
        raise ArgumentError, "position_scale must be finite and positive" unless position_scale.is_a?(Numeric) && position_scale.finite? && position_scale > 0
        %w[state_max_tokens question_option_max_tokens].each do |key|
          positive_integer!(self[:input][key], key)
          raise ArgumentError, "#{key} must be at least 4" if self[:input][key] < 4
        end
        raise ArgumentError, "truncation must be error or truncate" unless %w[error truncate].include?(self[:input]["truncation"])
        training = self[:training]
        %w[balance_labels balance_sources paired_sampling resample_negatives].each do |key|
          raise ArgumentError, "#{key} must be boolean" unless [true, false].include?(training[key])
        end
        if training["balance_sources"] && training["balance_labels"]
          raise ArgumentError, "Use balance_sources or balance_labels, not both"
        end
        if training["paired_sampling"] && (training["balance_sources"] || training["balance_labels"] || training["resample_negatives"])
          raise ArgumentError, "paired_sampling cannot combine with label/source balancing or negative resampling"
        end
        if training["paired_sampling"] && (!training["choice_microbatch"].is_a?(Integer) || !training["choice_microbatch"].even?)
          raise ArgumentError, "paired_sampling needs an even choice_microbatch"
        end
        raise ArgumentError, "seed must be a nonnegative integer" unless training["seed"].is_a?(Integer) && training["seed"] >= 0
        raise ArgumentError, "device must be auto, cuda or cpu" unless %w[auto cuda cpu].include?(training["device"])
        raise ArgumentError, "Only tested FP32 training is supported" unless training["precision"] == "float32"
        %w[choice_microbatch mlm_microbatch gradient_accumulation steps eval_every checkpoint_every].each do |key|
          positive_integer!(training[key], key)
        end
        %w[learning_rate grad_clip].each do |key|
          raise ArgumentError, "#{key} must be finite and positive" unless training[key].is_a?(Numeric) && training[key].finite? && training[key] > 0
        end
        raise ArgumentError, "invalid mask_probability" unless (0.0..1.0).cover?(training["mask_probability"]) && training["mask_probability"] > 0
        raise ArgumentError, "warmup_steps must be nonnegative" unless training["warmup_steps"].is_a?(Integer) && training["warmup_steps"] >= 0
        patience = training["early_stopping_patience"]
        raise ArgumentError, "early_stopping_patience must be nonnegative" unless patience.is_a?(Integer) && patience >= 0
        delta = training["selection_min_delta"]
        raise ArgumentError, "selection_min_delta must be finite and nonnegative" unless delta.is_a?(Numeric) && delta.finite? && delta >= 0
        raise ArgumentError, "weight_decay must be nonnegative" unless training["weight_decay"].is_a?(Numeric) && training["weight_decay"].finite? && training["weight_decay"] >= 0
        %w[gpu_memory_budget_mib candidate_chunk_size cache_entries].each { |key| positive_integer!(self[:runtime][key], key) }
        %w[patience trial_evaluations max_layers max_trials ffn_increment max_ffn_size].each { |key| positive_integer!(self[:growth][key], key) }
        raise ArgumentError, "unknown growth operation" unless %w[add_block widen_ffn].include?(self[:growth]["operation"])
        raise ArgumentError, "growth enabled must be boolean" unless [true, false].include?(self[:growth]["enabled"])
        delta = self[:growth]["min_delta"]
        raise ArgumentError, "growth min_delta must be finite and nonnegative" unless delta.is_a?(Numeric) && delta.finite? && delta >= 0
      end

      def positive_integer!(value, name)
        raise ArgumentError, "#{name} must be a positive integer" unless value.is_a?(Integer) && value > 0
      end
    end
  end
end
