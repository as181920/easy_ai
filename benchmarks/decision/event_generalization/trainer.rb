module DecisionEventGeneralization
  class Trainer < DecisionScaling::QuadTrainer
    def initialize(variant:, forbidden_states: [], migration: nil, **kwargs)
      @variant = variant.to_s
      raise ArgumentError, "Unknown v08 variant" unless %w[reference coverage event combined].include?(@variant)
      forbidden = forbidden_states.to_a.sort
      @signature = { "version" => 8, "variant" => @variant, "actor_weight" => 0.5, "event_weight" => 0.5,
        "forbidden_sha256" => Digest::SHA256.hexdigest(JSON.generate(forbidden)), "catalog_sha256" => Digest::SHA256.hexdigest(JSON.generate([Catalog::EVENTS, Catalog::POOLS, Catalog::FORMS])) }
      if kwargs[:restored] && kwargs[:restored].dig("training", "event_generalization") != @signature
        previous = kwargs[:restored].dig("training", "event_generalization")
        allowed = migration && migration.fetch("previous_signature") == previous && previous.except("forbidden_sha256") == @signature.except("forbidden_sha256") &&
          migration.fetch("new_forbidden_sha256") == @signature.fetch("forbidden_sha256") && migration.fetch("rendered_state_overlap").zero?
        raise ArgumentError, "v08 intervention changed on resume" unless allowed
      end
      rows = File.foreach(kwargs.fetch(:dataset).path).map { |line| JSON.parse(line) }
      @augmentation = Augmentation.new(rows: rows, forbidden: forbidden)
      actors = rows.to_h { |row| [row.fetch("id"), row.fetch("world").fetch("facts").map { |fact| fact.fetch("actor") }.uniq] }
      pools = Catalog::LANGUAGES.to_h { |language| [language, EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(language, "canonical")] }
      @renamer = EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: pools)
      @forbidden = forbidden.to_set
      super(**kwargs)
    end

    def save_checkpoint
      state["event_generalization"] = @signature
      super
    end

    private

      def loss_for(examples, seed:, teacher: false)
        covered = %w[coverage combined].include?(@variant)
        raw = @augmentation.call(examples, rng: Random.new(seed ^ 0xE818), coverage: covered)
        original = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
        rng = Random.new(seed ^ 0xA117)
        renamed = nil
        100.times do
          trial = @renamer.call(original, rng: rng)
          next if trial.any? { |row| @forbidden.include?(Catalog.material(row.state)) }
          renamed = trial
          break
        end
        raise "No reserved-material-free renamed batch" unless renamed
        record_coverage(raw, renamed) if teacher
        option_rng = Random.new(seed ^ 0xF177)
        examples = renamed.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.shuffle(random: option_rng))) }
        batch = @collator.call(examples, device: device)
        logits = model.call(batch)
        ce = Torch::NN::Functional.cross_entropy(logits, batch.fetch(:targets))
        actor = EasyAI::Decision::BindingPairLoss.call(logits, examples)
        event = %w[event combined].include?(@variant) ? PairLoss.call(logits, examples, axis: :event) : logits.sum * 0
        @grouped_input_tokens = @collator.input_token_counts.sum
        @batch_losses = { "choice_loss" => ce.item, "actor_pair_loss" => actor.item, "event_pair_loss" => event.item } if teacher
        ce + actor * 0.5 + event * 0.5
      end

      def record_coverage(raw, renamed)
        coverage = state["rendering_coverage"] ||= { "event_rows" => {}, "form_rows" => {}, "surface_hashes" => {} }
        raw.each do |row|
          event = "#{row.fetch('language')}/#{row.dig('world', 'query_event')}"
          form = "#{row.fetch('language')}/#{row.dig('world', 'form') || 0}"
          coverage.fetch("event_rows")[event] = coverage.fetch("event_rows").fetch(event, 0) + 1
          coverage.fetch("form_rows")[form] = coverage.fetch("form_rows").fetch(form, 0) + 1
        end
        renamed.each_slice(4) { |unit| coverage.fetch("surface_hashes")[Catalog.material(unit.first.state)] = true }
      end
  end
end
