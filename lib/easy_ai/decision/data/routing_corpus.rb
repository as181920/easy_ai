require "digest"
require "set"

module EasyAI
  module Decision
    module Data
      # Official partitions; original IDs and duplicate material stay together across languages.
      class RoutingCorpus
        attr_reader :exclusions

        def self.independent_rows(rows)
          rows.group_by { |row| [row.group_id, row.language] }.values.map do |group|
            group.min_by { |row| Digest::SHA256.hexdigest("release-independent-v01:#{row.id}") }
          end
        end

        def initialize(rows, historical_material:, train_material:, collator:, training_material: historical_material)
          @rows, @historical, @eligible, @collator = rows, historical_material, train_material, collator
          @training = training_material
          @parents, @exclusions = {}, Hash.new(0)
        end

        def splits
          @rows.each { |row| union(row.fetch("group_id"), NaturalCorpus.material(row.fetch("state"))) }
          components = @rows.group_by { |row| root(row.fetch("group_id")) }
          result = %w[train validation calibration test].to_h { |split| [split, []] }
          components.each do |group, rows|
            partitions = rows.map { |row| row.fetch("partition") }
            partition = %w[test dev train].find { |value| partitions.include?(value) }
            raise ArgumentError, "Unknown official partition" unless partition
            blocked = partition == "dev" ? @training : @historical
            if partition != "train" && rows.any? { |row| blocked.include?(NaturalCorpus.material(row.fetch("state"))) }
              @exclusions["historically_observed_#{partition}_components"] += 1
              next
            end
            rows = rows.select { |row| row.fetch("partition") == partition }
            conflicts = rows.group_by { |row| [NaturalCorpus.material(row.fetch("state")), row.fetch("language")] }
            if conflicts.values.any? { |items| items.map { |row| row.fetch("target") }.uniq.size > 1 }
              @exclusions["conflicting_components"] += 1
              next
            end
            split = partition == "dev" ? %w[validation calibration][Digest::SHA256.hexdigest("release-v01:#{group}").to_i(16) % 2] : partition
            rows.uniq { |row| [NaturalCorpus.material(row.fetch("state")), row.fetch("language")] }.each do |row|
              next if split == "train" && !@eligible.include?(NaturalCorpus.material(row.fetch("state")))
              next unless supported?(row)
              result.fetch(split) << row.merge("group_id" => "routing-component:#{group}")
            end
          end
          result
        end

        private

        def supported?(row)
          @collator.state_tokens(row.fetch("state"))
          row.fetch("options").each { |option| @collator.option_tokens(row.fetch("question"), option.fetch("text")) }
          true
        rescue ArgumentError => error
          raise unless error.message.include?("exceeds")
          @exclusions["over_length"] += 1
          false
        end

        def root(key)
          @parents[key] ||= key
          @parents[key] = root(@parents[key]) unless @parents[key] == key
          @parents[key]
        end

        def union(left, right)
          a, b = [root(left), root(right)].sort
          @parents[b] = a
        end
      end
    end
  end
end
