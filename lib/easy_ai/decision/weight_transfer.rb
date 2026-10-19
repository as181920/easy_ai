module EasyAI
  module Decision
    # Explicit own-parent transfer between compatible separate/joint recipes.
    class WeightTransfer
      def self.apply(parent:, config:, tokenizer:, seed:)
        source = parent.fetch(:model)
        unless tokenizer.fingerprint == parent.fetch(:tokenizer).fingerprint
          raise ArgumentError, "Transfer tokenizer fingerprint mismatch"
        end
        %w[vocab_size hidden_size encoder_layers attention_heads ffn_size ffn_sizes].each do |key|
          raise ArgumentError, "Transfer architecture mismatch: #{key}" unless config[:model][key] == source.config[:model][key]
        end
        Torch.manual_seed(seed)
        model = ChoiceModel.new(config)
        original, target = source.state_dict, model.state_dict
        missing = target.keys - original.keys
        unused = original.keys - target.keys
        allowed = config[:model]["encoding_mode"] == "joint" && source.config[:model]["encoding_mode"] == "separate"
        unless missing.empty? && (unused.empty? || (allowed && unused.all? { |key| key.start_with?("interactions.", "matching.") }))
          raise ArgumentError, "Unexpected transfer keys: missing=#{missing}, unused=#{unused}"
        end
        tensors = target.to_h do |key, value|
          raise ArgumentError, "Transfer shape mismatch: #{key}" unless value.shape == original.fetch(key).shape
          [key, original.fetch(key)]
        end
        model.load_state_dict(tensors)
        model.state_dict.each do |key, value|
          raise "Transfer value mismatch: #{key}" unless (value - original.fetch(key)).abs.max.item.zero?
        end
        manifest = { "copied" => tensors.keys.sort, "unused" => unused.sort, "new" => missing, "seed" => seed,
          "parameters" => model.parameter_count, "tokenizer_fingerprint" => tokenizer.fingerprint,
          "parent_weights_sha256" => parent.fetch(:weights_fingerprint) }
        [model, manifest]
      end
    end
  end
end
