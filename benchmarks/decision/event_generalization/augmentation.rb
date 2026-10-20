module DecisionEventGeneralization
  class Augmentation
    def initialize(rows:, forbidden: [])
      @rows = rows.to_h { |row| [row.fetch("id"), row] }
      @forbidden = forbidden.to_set
    end

    def call(examples, rng:, coverage: true)
      examples.each_slice(4).flat_map.with_index do |unit, index|
        unless unit.size == 4 && unit.map { |row| [row.language, row.source, row.group_id, row.state] }.uniq.size == 1
          raise ArgumentError, "Incomplete v08 augmentation unit"
        end
        original = unit.map do |example|
          row = @rows.fetch(example.id)
          row.merge("world" => row.fetch("world").merge("render_order" => row.fetch("world").fetch("render_order", row.fetch("id").split(":").last.to_i)))
        end
        next original unless coverage && index.odd?
        transformed = nil
        100.times do
          events = original.first.fetch("world").fetch("facts").map { |fact| fact.fetch("event") }.uniq
          mapping = events.zip(Catalog::POOLS.fetch("train").sample(events.size, random: rng)).to_h
          form = Catalog::FORMS.fetch("train").sample(random: rng)
          rows = original.map do |row|
            facts = row.fetch("world").fetch("facts").map { |fact| fact.merge("event" => mapping.fetch(fact.fetch("event"))) }
            Catalog.render(row, facts: facts, event: mapping.fetch(row.fetch("world").fetch("query_event")), form: form)
          end
          next if @forbidden.include?(Catalog.material(rows.first.fetch("state")))
          rows.each { |row| Corpus.gold!(row) }
          transformed = rows
          break
        end
        raise "No reserved-material-free coverage unit" unless transformed
        transformed
      end
    end
  end

  module PairLoss
    module_function

    def call(logits, examples, axis: :event)
      suffix = axis.to_s.capitalize
      raise ArgumentError, "Unknown binding axis" unless %w[Actor Event].include?(suffix)
      indices = examples.each_index.select { |index| examples[index].source.end_with?(suffix) }
      return logits.sum * 0 if indices.empty?
      selected = indices.map { |index| EasyAI::Decision::Data::Example.new(examples[index].to_h.merge("source" => "Factual-V08-Actor")) }
      scores = logits.index_select(0, Torch.tensor(indices, dtype: :int64, device: logits.device))
      EasyAI::Decision::BindingPairLoss.call(scores, selected)
    end
  end
end
