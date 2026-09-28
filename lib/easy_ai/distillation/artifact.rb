require "fileutils"

module EasyAI
  module Distillation
    # Complete, immutable JSONL artifact; only offsets are retained in memory.
    class Artifact
      attr_reader :manifest, :fingerprint, :path

      def initialize(path)
        @path = File.expand_path(path)
        @manifest = JSON.parse(File.read(File.join(@path, "manifest.json")))
        raise ArgumentError, "Expected complete distillation artifact v1" unless manifest["version"] == 1 && manifest["status"] == "complete"
        records = File.join(@path, "records.jsonl")
        raise ArgumentError, "Teacher records checksum mismatch" unless Digest::SHA256.file(records).hexdigest == manifest.fetch("records_sha256")
        @fingerprint = Fingerprint.call(manifest)
        @offsets = {}
        File.open(records, "rb") do |file|
          until file.eof?
            offset = file.pos
            row = JSON.parse(file.gets)
            id = row.fetch("identity").fetch("id")
            raise ArgumentError, "Duplicate teacher sample ID" if @offsets.key?(id)
            @offsets[id] = offset
          end
        end
        raise ArgumentError, "Teacher record count mismatch" unless @offsets.size == manifest.fetch("count")
      end

      def fetch(id)
        File.open(File.join(path, "records.jsonl"), "rb") do |file|
          file.seek(@offsets.fetch(id))
          JSON.parse(file.gets)
        end
      end

      def size
        @offsets.size
      end

      def self.write_json(path, value)
        temporary = "#{path}.part"
        File.write(temporary, JSON.pretty_generate(value) + "\n")
        File.rename(temporary, path)
      ensure
        FileUtils.rm_f(temporary) if temporary
      end
    end
  end
end
