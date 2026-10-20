module DecisionExpressionRetention
  class Augmentation
    def initialize(rows:)
      @rows = rows.to_h { |row| [row.fetch("id"), row] }
    end

    def call(examples, seed:, variant:)
      variant = variant.to_s
      raise ArgumentError, "Unknown v09 coverage" unless VARIANTS.include?(variant)
      event_rng = Random.new(seed ^ 0xE919)
      form_rng = Random.new(seed ^ 0xF919)
      examples.each_slice(4).flat_map.with_index do |unit, index|
        raise ArgumentError, "Incomplete coverage unit" unless unit.size == 4 && unit.map { |row| [row.language, row.source, row.group_id, row.state] }.uniq.size == 1
        original = unit.map do |example|
          row = @rows.fetch(example.id)
          row.merge("world" => row.fetch("world").merge("render_order" => row.fetch("world").fetch("render_order", row.fetch("id").split(":").last.to_i)))
        end
        events = original.first.dig("world", "facts").map { |fact| fact.fetch("event") }.uniq
        mapping = events.zip(Catalog::POOLS.fetch("train").sample(events.size, random: event_rng)).to_h
        form = Catalog::FORMS.fetch("train").sample(random: form_rng)
        change_events = index.odd? && %w[events both no_replay].include?(variant)
        change_form = variant == "no_replay" || (index.odd? && %w[expressions both].include?(variant))
        original.map do |row|
          facts = row.dig("world", "facts").map { |fact| change_events ? fact.merge("event" => mapping.fetch(fact.fetch("event"))) : fact }
          query = row.dig("world", "query_event")
          result = Catalog.render(row, facts: facts, event: change_events ? mapping.fetch(query) : query, form: change_form ? form : 0)
          Corpus.gold!(result)
          result
        end
      end
    end
  end

  class Trainer < DecisionScaling::QuadTrainer
    def initialize(variant:, forbidden_families: [], forbidden_states: [], **kwargs)
      @variant = variant.to_s
      raise ArgumentError, "Unknown v09 variant" unless VARIANTS.include?(@variant)
      @families = forbidden_families.to_set
      @states = forbidden_states.to_set
      @signature = { "version" => 9, "variant" => @variant, "actor_weight" => 0.5,
        "family_sha256" => Digest::SHA256.hexdigest(JSON.generate(@families.to_a.sort)), "state_sha256" => Digest::SHA256.hexdigest(JSON.generate(@states.to_a.sort)),
        "catalog_sha256" => Digest::SHA256.hexdigest(JSON.generate([Catalog::EVENTS, Catalog::POOLS, Catalog::FORMS])) }
      raise ArgumentError, "v09 intervention changed on resume" if kwargs[:restored] && kwargs[:restored].dig("training", "expression_retention") != @signature
      rows = File.foreach(kwargs.fetch(:dataset).path).map { |line| JSON.parse(line) }
      @augmentation = Augmentation.new(rows: rows)
      @pools = Catalog::LANGUAGES.to_h { |language| [language, EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(language, "canonical")] }
      super(**kwargs)
    end

    def save_checkpoint
      state["expression_retention"] = @signature
      super
    end

    def render_batch(examples, seed:)
      raw = @augmentation.call(examples, seed: seed, variant: @variant)
      rng = Random.new(seed ^ 0xA117)
      raw.each_slice(4).flat_map do |unit|
        original = unit.first.dig("world", "facts").map { |fact| fact.fetch("actor") }.uniq
        result = nil
        100.times do
          mapping = original.zip(@pools.fetch(unit.first.fetch("language")).sample(original.size, random: rng)).to_h
          trial = unit.map do |row|
            facts = row.dig("world", "facts").map { |fact| fact.merge("actor" => mapping.fetch(fact.fetch("actor"))) }
            Catalog.render(row, facts: facts, actor: mapping.fetch(row.dig("world", "actor")), form: row.dig("world", "form"))
          end
          next if @families.include?(Catalog.family(trial.first)) || @states.include?(Catalog.material(trial.first.fetch("state"))) || @states.include?(Digest::SHA256.hexdigest(trial.first.fetch("state")))
          result = trial
          break
        end
        raise "No reserved-family-free rename unit" unless result
        result
      end
    end

    private

      def loss_for(examples, seed:, teacher: false)
        rows = render_batch(examples, seed: seed)
        if teacher
          coverage = state["rendering_coverage"] ||= { "event_rows" => {}, "axis_form_rows" => {}, "surface_hashes" => {}, "families" => {} }
          rows.each do |row|
            event = "#{row.fetch('language')}/#{row.dig('world', 'query_event')}"
            form = "#{row.fetch('language')}/#{row.fetch('source').split('-').last}/#{row.dig('world', 'form')}"
            coverage["event_rows"][event] = coverage["event_rows"].fetch(event, 0) + 1
            coverage["axis_form_rows"][form] = coverage["axis_form_rows"].fetch(form, 0) + 1
          end
          rows.each_slice(4) do |unit|
            coverage["surface_hashes"][Catalog.material(unit.first.fetch("state"))] = true
            coverage["families"][Catalog.family(unit.first)] = true
          end
        end
        rng = Random.new(seed ^ 0xF177)
        examples = rows.map { |row| EasyAI::Decision::Data::Example.new(row.merge("options" => row.fetch("options").shuffle(random: rng))) }
        batch = @collator.call(examples, device: device)
        logits = model.call(batch)
        ce = Torch::NN::Functional.cross_entropy(logits, batch.fetch(:targets))
        actor = EasyAI::Decision::BindingPairLoss.call(logits, examples)
        @grouped_input_tokens = @collator.input_token_counts.sum
        @batch_losses = { "choice_loss" => ce.item, "actor_pair_loss" => actor.item } if teacher
        ce + actor * 0.5
      end
  end
end
