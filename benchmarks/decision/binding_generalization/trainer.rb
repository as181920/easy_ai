module DecisionBindingGeneralization
  class Trainer < DecisionScaling::QuadTrainer
    SIGNATURE = { "version" => 1, "pair_weight" => 0.5, "margin" => 1.0, "rename" => "per complete state unit, shared training identities, held-material rejection" }.freeze

    def initialize(variant:, **kwargs)
      @variant = variant.to_s
      raise ArgumentError, "Unknown binding intervention" unless %w[ce rename pair combined].include?(@variant)
      forbidden = kwargs.delete(:forbidden_states) { [] }.to_a.sort
      @signature = SIGNATURE.merge("variant" => @variant, "exclusions_sha256" => Digest::SHA256.hexdigest(JSON.generate(forbidden)))
      expected = @signature
      if kwargs[:restored] && kwargs[:restored].dig("training", "generalization") != expected
        raise ArgumentError, "Generalization intervention changed on resume"
      end
      if %w[rename combined].include?(@variant)
        raw = File.foreach(kwargs.fetch(:dataset).path).map { |line| JSON.parse(line) }
        actors = raw.to_h { |row| [row.fetch("id"), row.fetch("world").fetch("facts").map { |fact| fact.fetch("actor") }.uniq] }
        pools = %w[en-US zh-CN].to_h { |language| [language, EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(language, "canonical")] }
        @renamer = EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: pools, forbidden_states: forbidden)
      end
      super(**kwargs)
    end

    def save_checkpoint
      state["generalization"] = @signature
      super
    end

    private

      def loss_for(examples, seed:, teacher: false)
        examples = @renamer.call(examples, rng: Random.new(seed ^ 0xA117)) if @renamer
        rng = Random.new(seed ^ 0xF177)
        examples = examples.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.shuffle(random: rng))) }
        batch = @collator.call(examples, device: device)
        logits = model.call(batch)
        ce = Torch::NN::Functional.cross_entropy(logits, batch.fetch(:targets))
        coupled = %w[pair combined].include?(@variant)
        pair = EasyAI::Decision::BindingPairLoss.call(logits, examples) if coupled
        @grouped_input_tokens = @collator.input_token_counts.sum
        @batch_losses = { "choice_loss" => ce.item, "binding_pair_loss" => pair&.item || 0.0 } if teacher
        coupled ? ce + pair * SIGNATURE.fetch("pair_weight") : ce
      end
  end
end
