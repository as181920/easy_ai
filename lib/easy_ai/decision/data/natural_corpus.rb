require "digest"
require "set"

module EasyAI
  module Decision
    module Data
      # Components connect original IDs, parallel translations and identical material.
      # Splits use declared partitions and hash buckets, never target-dependent choices.
      class NaturalCorpus
        attr_reader :exclusions

        def self.material(text)
          Digest::SHA256.hexdigest(text.unicode_normalize(:nfkc).downcase.gsub(/\s+/, ""))
        end

        def initialize(rows, blocked:, collator:)
          raise ArgumentError, "Only official train/dev partitions may be split" unless rows.all? { |row| %w[train dev].include?(row.fetch("partition")) }
          @rows, @blocked, @collator = rows, blocked, collator
          @parents, @exclusions = {}, Hash.new(0)
          @materials = rows.to_h { |row| [row.object_id, self.class.material(row.fetch("state"))] }
        end

        def splits
          @rows.each { |row| union(row.fetch("group_id"), "material:#{material_for(row)}") }
          blocked_groups = @rows.filter_map { |row| root(row.fetch("group_id")) if @blocked.include?(material_for(row)) }.to_set
          labels = @rows.group_by { |row| input_key(row) }
          conflicting_groups = labels.values.select { |items| items.map { |row| row.fetch("target") }.uniq.size > 1 }
            .flat_map { |items| items.map { |row| root(row.fetch("group_id")) } }.to_set
          heldout_groups = @rows.select { |row| row.fetch("partition") == "dev" }.map { |row| root(row.fetch("group_id")) }.to_set
          result, seen = %w[train validation calibration test].to_h { |split| [split, []] }, Set.new
          @rows.each do |row|
            group = root(row.fetch("group_id"))
            if conflicting_groups.include?(group)
              @exclusions["#{row['source']}/conflicting_input_component"] += 1
              next
            end
            if blocked_groups.include?(group)
              @exclusions["#{row['source']}/historical_overlap"] += 1
              next
            end
            if heldout_groups.include?(group) && row.fetch("partition") != "dev"
              @exclusions["#{row['source']}/train_dev_component"] += 1
              next
            end
            key = input_key(row)
            unless seen.add?(key)
              @exclusions["#{row['source']}/duplicate_input"] += 1
              next
            end
            next unless within_limits?(row)
            row = row.merge("group_id" => "natural-component:#{group}")
            result.fetch(heldout_groups.include?(group) ? dev_split(group) : split(group)).push(row)
          end
          result
        end

        def split(group)
          bucket = Digest::SHA256.hexdigest("natural-v1:#{group}").to_i(16) % 100
          return "validation" if bucket < 5
          return "calibration" if bucket < 10
          return "test" if bucket < 15
          "train"
        end

        def dev_split(group)
          %w[validation calibration test].fetch(Digest::SHA256.hexdigest("natural-dev-v1:#{group}").to_i(16) % 3)
        end

        private

        def material_for(row)
          @materials.fetch(row.object_id)
        end

        def input_key(row)
          [material_for(row), row.fetch("question"), row.fetch("source"), row.fetch("language")]
        end

        def within_limits?(row)
          @collator.state_tokens(row.fetch("state"))
          row.fetch("options").each { |option| @collator.option_tokens(row.fetch("question"), option.fetch("text")) }
          true
        rescue ArgumentError => error
          raise unless error.message.include?("exceeds")
          @exclusions["#{row['source']}/over_length"] += 1
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
