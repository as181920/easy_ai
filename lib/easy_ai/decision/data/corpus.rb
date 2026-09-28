require "json"
require "fileutils"
require "digest"

module EasyAI
  module Decision
    module Data
      class Corpus
        # Accept local FineWeb2 JSONL exports or any public JSONL with text.
        # Hash normalized text to keep exact duplicate documents in one split.
        def self.prepare(input:, output:, language: "und", limit: nil)
          raise ArgumentError, "Output already exists" if File.exist?(output)
          FileUtils.mkdir_p(output)
          seen, counts = Set.new, Hash.new(0)
          writers = %w[train validation].to_h { |split| [split, File.open(File.join(output, "#{split}.jsonl"), "w")] }
          begin
            File.foreach(input).with_index do |line, index|
              break if limit && counts.values.sum >= limit
              next if line.strip.empty?
              row = JSON.parse(line)
              text = row.fetch("text")
              next if text.strip.empty?
              normalized = text.unicode_normalize(:nfc).gsub(/\s+/, " ").strip
              digest = Digest::SHA256.hexdigest(normalized)
              next unless seen.add?(digest)
              split = digest.to_i(16) % 20 == 0 ? "validation" : "train"
              writers.fetch(split).puts(JSON.generate("id" => row.fetch("id", index.to_s), "group_id" => "text:#{digest}",
                "language" => row.fetch("language", language), "source" => row.fetch("source", File.basename(input)), "text" => text))
              counts[split] += 1
            end
          ensure
            writers.each_value(&:close)
          end
          manifest = { "input_sha256" => Digest::SHA256.file(input).hexdigest, "counts" => counts,
            "split_policy" => "normalized exact text SHA256 modulo 20; 95% train, 5% validation; not near-duplicate removal" }
          File.write(File.join(output, "manifest.json"), JSON.pretty_generate(manifest))
          manifest
        end
      end
    end
  end
end
