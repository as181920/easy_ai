require "csv"
require "zlib"
require "rubygems/package"

module EasyAI
  module Decision
    module Data
      # Dataset labels and fixed class descriptions; no text-classification rules.
      class NaturalAdapter
        NEWS_LABELS = ["World news", "Sports news", "Business news", "Science and technology news"].freeze
        EMOTIONS = %w[sadness joy love anger fear surprise].freeze
        SCENARIOS = {
          "alarm" => ["闹钟", "alarms"], "audio" => ["音频设置", "audio settings"],
          "calendar" => ["日历与预约", "calendar and appointments"], "cooking" => ["烹饪", "cooking"],
          "datetime" => ["日期与时间", "date and time"], "email" => ["电子邮件", "email"],
          "general" => ["一般对话", "general conversation"], "iot" => ["智能家居设备", "smart home devices"],
          "lists" => ["清单", "lists"], "music" => ["音乐信息与管理", "music information and management"],
          "news" => ["新闻", "news"], "play" => ["播放媒体", "media playback"],
          "qa" => ["知识问答", "factual questions and answers"], "recommendation" => ["推荐建议", "recommendations"],
          "social" => ["社交媒体", "social media"], "takeaway" => ["外卖", "takeaway food"],
          "transport" => ["交通出行", "transport"], "weather" => ["天气", "weather"]
        }.freeze

        def self.news(path)
          return enum_for(:news, path) unless block_given?
          CSV.foreach(path, encoding: "UTF-8").with_index do |row, index|
            raise ArgumentError, "Expected label/title/description" unless row.size == 3
            label = Integer(row.fetch(0), 10) - 1
            raise ArgumentError, "Unknown news label" unless (0...NEWS_LABELS.size).cover?(label)
            yield build("AG-News", "en-US", index.to_s, [row.fetch(1), row.fetch(2)].join(" "),
              "What is the topic of this news article?", NEWS_LABELS.each_with_index.to_h { |text, i| [i.to_s, text] }, label.to_s, "train")
          end
        end

        def self.massive(path, partitions: %w[train dev])
          unless partitions.is_a?(Array) && !partitions.empty? && (partitions - %w[train dev test]).empty?
            raise ArgumentError, "Unknown MASSIVE partitions"
          end
          return enum_for(:massive, path, partitions: partitions) unless block_given?
          Zlib::GzipReader.open(path) do |gzip|
            Gem::Package::TarReader.new(gzip) do |tar|
              tar.each do |entry|
                locale = File.basename(entry.full_name, ".jsonl")
                next unless entry.file? && %w[en-US zh-CN].include?(locale)
                SemanticAdapter.each_tar_line(entry) do |line|
                  row = JSON.parse(line)
                  raise ArgumentError, "Unexpected MASSIVE partition" unless %w[train dev test].include?(row.fetch("partition"))
                  next unless partitions.include?(row.fetch("partition"))
                  yield build("MASSIVE-Scenario", locale, row.fetch("id").to_s, row.fetch("utt"),
                    locale == "zh-CN" ? "这项请求属于哪个领域？" : "Which domain does this request belong to?",
                    SCENARIOS.transform_values { |texts| texts.fetch(locale == "zh-CN" ? 0 : 1) }, row.fetch("scenario"), row.fetch("partition"))
                end
              end
            end
          end
        end

        def self.emotion(path)
          return enum_for(:emotion, path) unless block_given?
          wrapped = JSON.parse(File.read(path)).fetch("rows")
          raise ArgumentError, "Truncated emotion cells" if wrapped.any? { |item| item.fetch("truncated_cells", []).any? }
          wrapped.each do |item|
            row = item.fetch("row")
            label = row.fetch("label")
            target = label.is_a?(Integer) ? label.to_s : EMOTIONS.index(label)&.to_s
            raise ArgumentError, "Unknown emotion label" unless target && (0...EMOTIONS.size).cover?(target.to_i)
            yield build("Emotion", "en-US", item.fetch("row_idx").to_s, row.fetch("text"), "Which emotion does the text express?",
              EMOTIONS.each_with_index.to_h { |text, i| [i.to_s, text] }, target, "heldout")
          end
        end

        def self.build(source, language, origin, state, question, labels, target, partition)
          options = labels.map { |id, text| { "id" => id, "text" => text } }
          row = { "id" => "natural:#{source}:#{origin}:#{language}", "group_id" => "natural:#{source}:#{origin}", "source" => source,
            "language" => language, "state" => state, "question" => question, "options" => options, "target" => target, "partition" => partition }
          Example.new(row)
          row
        end
      end
    end
  end
end
