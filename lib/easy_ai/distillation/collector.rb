require "set"

module EasyAI
  module Distillation
    class Collector
      def initialize(teacher:, adapter:, output:, purpose:, progress: $stderr)
        raise ArgumentError, "Purpose must be train or development" unless %w[train development].include?(purpose)
        @teacher, @adapter, @output, @purpose, @progress = teacher, adapter, File.expand_path(output), purpose, progress
      end

      def run(dataset)
        contract = { "version" => 1, "purpose" => @purpose, "source_sha256" => dataset.fingerprint,
          "teacher" => @teacher.signature, "adapter" => @adapter.signature }
        FileUtils.mkdir_p(@output)
        File.open(File.join(@output, "collector.lock"), "w") do |lock|
          raise ArgumentError, "Another collector is running" unless lock.flock(File::LOCK_EX | File::LOCK_NB)
          contract_path = File.join(@output, "contract.json")
          if File.exist?(contract_path)
            raise ArgumentError, "Collection configuration or data changed; use a new output" unless JSON.parse(File.read(contract_path)) == contract
          else
            Artifact.write_json(contract_path, contract)
          end
          return Artifact.new(@output) if File.exist?(File.join(@output, "manifest.json"))
          collect(dataset, contract)
        end
      end

      private

      def collect(dataset, contract)
        cache = File.join(@output, "responses")
        FileUtils.mkdir_p(cache)
        ids = Set.new
        temporary = File.join(@output, "records.jsonl.part")
        File.open(temporary, "w") do |file|
          dataset.each_with_index do |example, index|
            reply = nil
            identity = @adapter.identity(example)
            raise ArgumentError, "Duplicate sample ID" unless ids.add?(identity.fetch("id"))
            request = @adapter.request(example)
            key = Fingerprint.call("identity" => identity, "request" => request)
            cached = File.join(cache, "#{key}.json")
            record = if File.exist?(cached)
              JSON.parse(File.read(cached))
                     else
              reply = @teacher.call(request)
              value = { "identity" => identity, "request_sha256" => Fingerprint.call(request), "reply" => reply,
                "supervision" => @adapter.parse(example, reply) }
              Artifact.write_json(cached, value)
              value
                     end
            unless record["identity"] == identity && record["request_sha256"] == Fingerprint.call(request) &&
                record["supervision"] == @adapter.parse(example, record.fetch("reply"))
              raise ArgumentError, "Invalid cached teacher response"
            end
            file.puts(JSON.generate(record))
            @progress&.puts("Distillation collect #{index + 1}/#{dataset.size}")
          rescue StandardError => error
            Artifact.write_json(File.join(@output, "failure.json"), { "index" => index, "error_class" => error.class.name,
              "message" => error.message, "reply" => reply })
            raise
          end
        end
        records = File.join(@output, "records.jsonl")
        File.rename(temporary, records)
        Artifact.write_json(File.join(@output, "manifest.json"), contract.merge("status" => "complete",
          "count" => ids.size, "records_sha256" => Digest::SHA256.file(records).hexdigest))
        FileUtils.rm_f(File.join(@output, "failure.json"))
        Artifact.new(@output)
      end
    end
  end
end
