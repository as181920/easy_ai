require "digest"
require "json"
require "fileutils"

module EasyAI
  module Decision
    module Data
      class SemanticCorpus
        SPLITS = %w[train validation calibration test].freeze

        def initialize(rows, seed: 1337)
          @rows, @seed = rows, seed
          @parents = {}
          @keys = rows.map do |row|
            origin = digest("#{row.fetch(:source)}:#{row.fetch(:origin)}")
            # Group short generic answers by question; group long repeated passages across sources.
            state = normalized(row.fetch(:state))
            material = digest(state.length >= 40 ? "material:#{state}" : "pair:#{state}:#{normalized(row.fetch(:question))}")
            union(origin, material)
            origin
          end
        end

        def split_rows
          held_out = @rows.each_index.filter_map { |i| root(@keys[i]) if @rows[i][:partition] == "dev" }.to_set
          @rows.each_with_index.filter_map do |row, index|
            group = root(@keys[index])
            # A source training duplicate of public dev must not migrate into training or test.
            next if held_out.include?(group) && row[:partition] != "dev"
            bucket = Digest::SHA256.hexdigest("#{@seed}:#{group}").to_i(16) % 100
            split = if held_out.include?(group)
              "test"
                    elsif bucket < 5
              "validation"
                    elsif bucket < 10
              "calibration"
                    else
              "train"
                    end
            [row, "semantic:#{group}", split]
          end
        end

        def write(output:, config:, tokenizer:, limit: 100, train_limit: nil)
          raise ArgumentError, "Output exists: #{output}" if File.exist?(output)
          raise ArgumentError, "limit must be positive" unless limit.is_a?(Integer) && limit > 0
          raise ArgumentError, "train_limit must be positive" if train_limit && (!train_limit.is_a?(Integer) || train_limit <= 0)
          rows = split_rows
          train_rows = rows.select { |_row, _group, split| split == "train" }
          tokenizer.train(train_rows.flat_map { |row, _group, _split| [row[:state], row[:question], *row[:options].values] },
            vocab_size: config[:model]["vocab_size"])
          FileUtils.mkdir_p(output)
          tokenizer.save(File.join(output, "tokenizer.json"))
          collator = Collator.new(tokenizer: tokenizer, config: config.with(input: { truncation: "error" }))
          writers = SPLITS.to_h { |split| [split, File.open(File.join(output, "#{split}.jsonl"), "w")] }
          corpora = %w[train validation].to_h { |split| [split, File.open(File.join(output, split == "train" ? "corpus.jsonl" : "corpus-validation.jsonl"), "w")] }
          counts, skipped, selected, seen = Hash.new(0), Hash.new(0), Hash.new { |h, k| h[k] = Set.new }, Set.new
          begin
            rows.sort_by { |row, group, split| [split, row[:source], digest("#{@seed}:#{group}"), row[:question]] }.each do |row, group, split|
              key = "#{split}/#{row[:source]}"
              identity = digest([row[:source], row[:state], row[:question], row[:target]].join("\n"))
              next unless seen.add?(identity)
              budget = split == "train" ? train_limit : limit
              next if budget && selected[key].size >= budget && !selected[key].include?(group)
              rng = Random.new(identity.to_i(16))
              options = row[:options].map { |id, text| { "id" => id, "text" => text } }.shuffle(random: rng)
              example = Example.new({ id: identity, group_id: group, language: row[:language], source: row[:source],
                state: row[:state], question: row[:question], target: row[:target], options: options })
              begin
                collator.state_tokens(example.state)
                example.options.each { |option| collator.option_tokens(example.question, option.fetch("text")) }
              rescue ArgumentError => error
                raise unless error.message.include?("exceeds")
                skipped[key] += 1
                next
              end
              selected[key] << group
              writers.fetch(split).puts(JSON.generate(example.to_h))
              if corpora[split]
                [example.state, example.question].each_with_index do |text, part|
                  corpora[split].puts(JSON.generate(id: "#{identity}:#{part}", group_id: group, language: example.language,
                    source: example.source, text: text))
                end
              end
              counts[key] += 1
            end
          ensure
            (writers.values + corpora.values).each(&:close)
          end
          datasets = SPLITS.map { |split| Dataset.new(File.join(output, "#{split}.jsonl")) }
          Dataset.assert_disjoint!(*datasets)
          manifest = { "seed" => @seed, "rows" => counts, "groups" => selected.transform_values(&:size),
            "over_length_skipped" => skipped, "raw_rows" => @rows.size, "deduplicated_split_rows" => rows.size,
            "input" => config[:input], "tokenizer_fingerprint" => tokenizer.fingerprint,
            "files_sha256" => Dir.glob(File.join(output, "*.jsonl")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
            "tokenizer_policy" => "native BPE fitted only on eligible training source text, before length filtering and group cap",
            "split_policy" => "Public dev is local test. Training groups hash into 90% train, 5% validation, 5% calibration. Shared material/origin components never cross splits.",
            "scope" => "Noncommercial learning experiment. DuReader uses annotated answer as state; not document retrieval or answer generation." }
          File.write(File.join(output, "manifest.json"), JSON.pretty_generate(manifest))
          manifest
        end

        private

        def normalized(text)
          text.unicode_normalize(:nfkc).gsub(/\s+/, "").downcase
        end

        def digest(text)
          Digest::SHA256.hexdigest(text)
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
