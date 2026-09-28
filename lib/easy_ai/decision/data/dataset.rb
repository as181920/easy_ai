require "json"
require "digest"

module EasyAI
  module Decision
    module Data
      # Index offsets, not complete documents, so corpus size need not fit RAM.
      class Dataset
        include Enumerable
        attr_reader :path, :fingerprint, :kind

        def initialize(path, kind: :choice, require_target: true)
          @path, @kind = File.expand_path(path), kind
          @require_target = require_target
          @offsets = []
          File.open(@path, "rb") do |file|
            until file.eof?
              offset = file.pos
              line = file.gets
              @offsets << offset unless line.strip.empty?
            end
          end
          raise ArgumentError, "Empty dataset: #{path}" if @offsets.empty?
          @fingerprint = Digest::SHA256.file(@path).hexdigest
        end

        def size
          @offsets.length
        end

        def [](index)
          line = File.open(path, "rb") { |file| file.seek(@offsets.fetch(index)); file.gets }
          parse(line)
        end

        def each
          return enum_for(:each) unless block_given?
          File.foreach(path) { |line| yield parse(line) unless line.strip.empty? }
        end

        def groups
          map { |item| item.is_a?(Example) ? item.group_id : item.fetch("group_id", item.fetch("id")) }.to_set
        end

        def self.assert_disjoint!(*datasets)
          datasets.compact.combination(2).each do |left, right|
            overlap = left.groups & right.groups
            raise ArgumentError, "Dataset split leakage: #{overlap.first}" unless overlap.empty?
          end
        end

        private

        def parse(line)
          row = JSON.parse(line.force_encoding("UTF-8"))
          if kind == :choice
            Example.new(row, require_target: @require_target)
          else
            text = row.fetch("text")
            raise ArgumentError, "Empty corpus text" unless text.is_a?(String) && !text.strip.empty?
            row
          end
        end
      end
    end
  end
end
