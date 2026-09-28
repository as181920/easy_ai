require "json"
require "digest"
require "fileutils"
require "zlib"
require "rubygems/package"

module EasyAI
  module Decision
    module Data
      module Adapters
        class Massive
          URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz".freeze
          LICENSE = "CC-BY-4.0".freeze
          DEFAULT_LOCALES = %w[zh-CN en-US ja-JP es-ES ar-SA].freeze

          # Read only explicitly selected files; no archive paths are extracted.
          def self.prepare(archive:, output:, locales: DEFAULT_LOCALES, candidates: 8, limit: nil, train_limit: nil, seed: 1337, descriptions: nil)
            raise ArgumentError, "candidates must be >=2" unless candidates >= 2
            raise ArgumentError, "limit must be positive" if limit && limit <= 0
            raise ArgumentError, "train_limit must be positive" if train_limit && train_limit <= 0
            raise ArgumentError, "Output exists: #{output}" if File.exist?(output)
            rows = {}
            license_text = nil
            Zlib::GzipReader.open(archive) do |gzip|
              Gem::Package::TarReader.new(gzip) do |tar|
                tar.each do |entry|
                  next unless entry.file?
                  filename = File.basename(entry.full_name)
                  if filename == "LICENSE"
                    license_text = entry.read
                  elsif locales.include?(File.basename(filename, ".jsonl")) && filename.end_with?(".jsonl")
                    rows[File.basename(filename, ".jsonl")] = entry.read.lines.map { |line| JSON.parse(line) }
                  end
                end
              end
            end
            raise ArgumentError, "Missing locales: #{locales - rows.keys}" unless (locales - rows.keys).empty?
            all = rows.values.flatten
            labels = all.select { |r| r.fetch("partition") == "train" }.map { |r| r.fetch("intent") }.uniq.sort
            raise ArgumentError, "Too many requested candidates" if candidates > labels.length
            descriptions ||= labels.to_h { |label| [label, label.tr("_", " ")] }
            raise ArgumentError, "Descriptions missing labels" unless (labels - descriptions.keys).empty?
            FileUtils.mkdir_p(output)
            counts = Hash.new { |h, k| h[k] = Hash.new(0) }
            groups = Hash.new { |h, k| h[k] = Set.new }
            writers = %w[train validation calibration test].to_h { |split| [split, File.open(File.join(output, "#{split}.jsonl"), "w")] }
            corpus = File.open(File.join(output, "corpus.jsonl"), "w")
            corpus_validation = File.open(File.join(output, "corpus-validation.jsonl"), "w")
            begin
              locales.each do |locale|
                rows.fetch(locale).sort_by { |r| Digest::SHA256.hexdigest("#{seed}:#{r.fetch('id')}") }.each do |row|
                  group_id = "massive:#{row.fetch('id')}"
                  split = case row.fetch("partition")
                          when "train" then "train"
                          when "test" then "test"
                          when "dev" then Digest::SHA256.hexdigest(group_id).to_i(16).even? ? "validation" : "calibration"
                          else raise ArgumentError, "Unknown MASSIVE partition"
                          end
                  split_limit = split == "train" && train_limit ? train_limit : limit
                  next if split_limit && counts[split][locale] >= split_limit
                  rng = Random.new(Digest::SHA256.hexdigest("#{seed}:#{group_id}").to_i(16))
                  target = row.fetch("intent")
                  chosen = ([target] + (labels - [target]).sample(candidates - 1, random: rng)).shuffle(random: rng)
                  example = Example.new({ "id" => "#{group_id}:#{locale}", "group_id" => group_id,
                    "language" => locale, "source" => "MASSIVE-1.1", "state" => row.fetch("utt"),
                    "question" => "Intent?", "options" => chosen.map { |id| { "id" => id, "text" => descriptions.fetch(id) } }, "target" => target })
                  writers.fetch(split).puts(JSON.generate(example.to_h))
                  if %w[train validation].include?(split)
                    writer = split == "train" ? corpus : corpus_validation
                    writer.puts(JSON.generate("id" => example.id, "group_id" => group_id, "language" => locale,
                      "source" => "MASSIVE-1.1", "text" => example.state))
                  end
                  counts[split][locale] += 1
                  groups[split] << group_id
                end
              end
            ensure
              writers.each_value(&:close)
              corpus.close
              corpus_validation.close
            end
            groups.values.combination(2).each { |a, b| raise "Parallel sample split leakage" unless (a & b).empty? }
            File.write(File.join(output, "LICENSE"), license_text || "MASSIVE dataset: #{LICENSE}. Copyright Amazon.com, Inc. or its affiliates.\n")
            manifest = { "source" => URL, "license" => LICENSE, "archive_sha256" => Digest::SHA256.file(archive).hexdigest,
              "locales" => locales, "candidates" => candidates, "seed" => seed, "limit_per_locale_per_split" => limit,
              "train_limit_per_locale" => train_limit || limit,
              "counts" => counts, "split_policy" => "official train/test; dev grouped by original ID into validation/calibration",
              "candidate_policy" => "target plus sampled negatives; NOT the full 60-class benchmark",
              "description_policy" => "English intent labels unless supplied; evaluates cross-language choices" }
            File.write(File.join(output, "manifest.json"), JSON.pretty_generate(manifest))
            manifest
          end
        end
      end
    end
  end
end
