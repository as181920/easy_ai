require "digest"
require "set"

module EasyAI
  module Decision
    # Training augmentation only. Metadata identifies spans, never an answer.
    class EntityRenaming
      def initialize(actors_by_id:, pools:, forbidden_states: [])
        @actors_by_id = actors_by_id
        @pools = pools
        @forbidden_states = forbidden_states.to_set
      end

      def call(examples, rng:)
        examples.each_slice(4).flat_map do |unit|
          actors = @actors_by_id.fetch(unit.first.id).uniq
          unless unit.size == 4 && unit.map { |row| [row.language, row.group_id, row.state] }.uniq.size == 1 && unit.all? { |row| @actors_by_id.fetch(row.id).uniq == actors }
            raise ArgumentError, "Incomplete renaming unit"
          end
          pool = @pools.fetch(unit.first.language)
          raise ArgumentError, "Insufficient distinct rename identities" if pool.size < actors.size || pool.uniq.size != pool.size
          mapping = nil
          100.times do
            candidate = actors.zip(pool.sample(actors.size, random: rng)).to_h
            state = replace(unit.first.state, candidate)
            next if @forbidden_states.include?(Digest::SHA256.hexdigest(state))
            mapping = candidate
            break
          end
          raise ArgumentError, "No leakage-free rename found" unless mapping
          unit.map do |row|
            Data::Example.new(row.to_h.merge("state" => replace(row.state, mapping), "question" => replace(row.question, mapping)))
          end
        end
      end

      private

        def replace(text, mapping)
          text.gsub(Regexp.union(mapping.keys.sort_by { |key| -key.length })) { |match| mapping.fetch(match) }
        end
    end
  end
end
