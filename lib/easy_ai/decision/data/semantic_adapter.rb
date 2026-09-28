require "json"
require "zlib"
require "rubygems/package"
require "open3"

module EasyAI
  module Decision
    module Data
      # Converts only public labeled train/dev files; hidden test files are never read.
      class SemanticAdapter
        def self.each(directory)
          return enum_for(:each, directory) unless block_given?
          %w[train dev].each do |split|
            File.foreach(File.join(directory, "ocnli-#{split}.jsonl")) do |line|
              row = JSON.parse(line)
              next unless %w[entailment contradiction neutral].include?(row["label"])
              yield({ source: "OCNLI", language: "zh-CN", partition: split, origin: row.fetch("prem_id"),
                state: row.fetch("sentence1"), question: "根据上下文，以下陈述是否成立：#{row.fetch('sentence2')}",
                target: { "entailment" => "yes", "contradiction" => "no", "neutral" => "unknown" }.fetch(row["label"]),
                options: { "yes" => "成立", "no" => "不成立", "unknown" => "信息不足，无法判断" } })
            end
          end
          Zlib::GzipReader.open(File.join(directory, "dureader-yesno.tar.gz")) do |gzip|
            Gem::Package::TarReader.new(gzip) do |tar|
              tar.each do |entry|
                split = { "train.json" => "train", "dev.json" => "dev" }[File.basename(entry.full_name)]
                next unless entry.file? && split
                each_tar_line(entry) do |line|
                  row = JSON.parse(line)
                  next unless %w[Yes No Depends].include?(row["yesno_answer"])
                  yield({ source: "DuReader-YesNo", language: "zh-CN", partition: split, origin: row.fetch("question"),
                    state: row.fetch("answer"), question: row.fetch("question"),
                    target: { "Yes" => "yes", "No" => "no", "Depends" => "depends" }.fetch(row["yesno_answer"]),
                    options: { "yes" => "是", "no" => "否", "depends" => "视情况而定" } })
                end
              end
            end
          end
          %w[train val].each do |split|
            Open3.popen3("unzip", "-p", File.join(directory, "boolq.zip"), "BoolQ/#{split}.jsonl") do |input, stdout, stderr, process|
              input.close
              errors = Thread.new { stderr.read }
              stdout.each_line do |line|
                row = JSON.parse(line)
                label = row.fetch("label")
                raise ArgumentError, "Unknown BoolQ label" unless [true, false].include?(label)
                yield({ source: "BoolQ", language: "en-US", partition: split == "val" ? "dev" : "train", origin: row.fetch("passage"),
                  state: row.fetch("passage"), question: row.fetch("question"), target: label ? "yes" : "no",
                  options: { "yes" => "yes", "no" => "no" } })
              end
              raise ArgumentError, "Cannot read BoolQ archive: #{errors.value}" unless process.value.success?
              errors.value
            end
          end
        end

        def self.each_tar_line(entry)
          buffer = "".b
          until entry.eof?
            buffer << entry.read(64 * 1024)
            while (newline = buffer.index("\n"))
              yield buffer.slice!(0..newline).force_encoding("UTF-8")
            end
          end
          yield buffer.force_encoding("UTF-8") unless buffer.empty?
        end
      end
    end
  end
end
