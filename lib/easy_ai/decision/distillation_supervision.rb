module EasyAI
  module Decision
    class DistillationSupervision
      attr_reader :signature

      def initialize(artifact:, dataset:, weight: 1.0, temperature: 1.0)
        unless [weight, temperature].all? { |value| value.is_a?(Numeric) && value.finite? && value > 0 }
          raise ArgumentError, "Teacher weight and temperature must be finite and positive"
        end
        @artifact = artifact.is_a?(Distillation::Artifact) ? artifact : Distillation::Artifact.new(artifact)
        @adapter, @weight, @temperature = DistillationAdapter.from_signature(@artifact.manifest.fetch("adapter")), weight, temperature
        manifest = @artifact.manifest
        unless manifest["purpose"] == "train" && manifest["source_sha256"] == dataset.fingerprint && manifest["adapter"] == @adapter.signature
          raise ArgumentError, "Teacher artifact must match this training dataset and adapter"
        end
        raise ArgumentError, "Teacher artifact count differs from training data" unless @artifact.size == dataset.size
        kinds = dataset.map { |example| signal(example).fetch("kind") }.uniq
        raise ArgumentError, "Do not mix hard and soft teacher signals in one artifact" unless kinds.size == 1
        @kind = kinds.first
        raise ArgumentError, "Hard teacher labels require temperature=1" if @kind == "label" && temperature != 1
        @signature = { "artifact_sha256" => @artifact.fingerprint, "weight" => weight, "temperature" => temperature }
      end

      def loss(logits, examples)
        distributions = examples.map do |example|
          value = signal(example)
          example.options.map do |option|
            value["kind"] == "label" ? (option.fetch("id") == value.fetch("target") ? 1.0 : 0.0) : value.fetch("probabilities").fetch(option.fetch("id"))
          end
        end
        # One-hot KL at T=1 is hard pseudo-label CE; temperature applies only to soft targets.
        Distillation::Losses::SoftTargets.call(logits, distributions, temperature: @kind == "label" ? 1.0 : @temperature) * @weight
      end

      private

      def signal(example)
        record = @artifact.fetch(example.id)
        raise ArgumentError, "Teacher input identity mismatch" unless record.fetch("identity") == @adapter.identity(example)
        value = record.fetch("supervision")
        if value["kind"] == "label"
          raise ArgumentError, "Teacher target not in candidates" unless example.options.any? { |option| option["id"] == value["target"] }
        elsif value["kind"] == "candidate_probabilities"
          @adapter.parse(example, value.merge("scoring_protocol" => "artifact_validation"))
        else
          raise ArgumentError, "Invalid teacher supervision"
        end
        value
      end
    end
  end
end
