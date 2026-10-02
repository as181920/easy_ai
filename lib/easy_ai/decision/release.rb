require "json"
require "fileutils"
require "digest"

module EasyAI
  module Decision
    # A locally published, measured model bundle; no business actions are executed.
    class Release
      attr_reader :predictor, :metadata

      def self.publish(run:, output:, preview: false)
        raise ArgumentError, "Release destination exists" if File.exist?(output)
        acceptance = JSON.parse(File.read(File.join(run, "acceptance.json")))
        policy = JSON.parse(File.read(File.join(run, "policy.json")))
        protocol = JSON.parse(File.read(File.join(run, "protocol.json")))
        raise ArgumentError, "Release did not pass acceptance" unless preview || acceptance.fetch("passed")
        unless acceptance.fetch("policy_sha256") == Digest::SHA256.file(File.join(run, "policy.json")).hexdigest &&
            acceptance.fetch("protocol_sha256") == Digest::SHA256.file(File.join(run, "protocol.json")).hexdigest
          raise ArgumentError, "Release acceptance provenance changed"
        end
        validate_acceptance!(acceptance, protocol, preview: preview)
        runtime = JSON.parse(File.read(File.join(run, "runtime.json")))
        raise ArgumentError, "Release runtime checks failed" unless runtime.fetch("passed")
        loaded = Checkpoint.load(policy.fetch("checkpoint"))
        raise ArgumentError, "Accepted checkpoint changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
        raise ArgumentError, "Runtime checkpoint changed" unless runtime.fetch("weights_sha256") == loaded.fetch(:weights_fingerprint)
        raise ArgumentError, "Checkpoint temperature changed" unless loaded.fetch(:metadata).dig("calibration", "temperature") == policy.fetch("temperature")
        FileUtils.mkdir_p(output)
        FileUtils.cp_r(loaded.fetch(:path), File.join(output, "checkpoint"))
        FileUtils.cp(File.join(run, "acceptance.json"), File.join(output, "acceptance.json"))
        FileUtils.cp(File.join(run, "runtime.json"), File.join(output, "runtime.json"))
        info = { "version" => "0.1", "profile" => protocol.fetch("profile"), "languages" => policy.fetch("languages"),
          "temperature" => policy.fetch("temperature"), "weights_sha256" => policy.fetch("weights_sha256"),
          "candidate_chunk_size" => 18, "status" => preview ? "preview" : "stable",
          "acceptance_passed" => acceptance.fetch("passed"),
          "scope" => protocol.fetch("scope"), "requirements" => protocol.fetch("requirements"),
          "source" => protocol.fetch("source"), "parent_sha256" => protocol.fetch("parent_sha256"),
          "protocol_sha256" => acceptance.fetch("protocol_sha256"),
          "license_scope" => "MASSIVE CC-BY-4.0; own parent inherits its earlier public-source conditions. See docs/decision/natural.md." }
        Checkpoint.atomic_json(File.join(output, "release.json"), info)
        files = %w[release.json acceptance.json runtime.json] + Dir.children(File.join(output, "checkpoint")).map { |name| "checkpoint/#{name}" }
        hashes = files.to_h { |name| [name, Digest::SHA256.file(File.join(output, name)).hexdigest] }
        Checkpoint.atomic_json(File.join(output, "bundle-manifest.json"), hashes)
        output
      end

      def self.validate_acceptance!(acceptance, protocol, preview:)
        unless protocol.fetch("requirements") == ReleasePolicy::REQUIREMENTS && protocol.fetch("languages").sort == %w[en-US zh-CN]
          raise ArgumentError, "Unknown release requirements or languages"
        end
        passed = true
        protocol.fetch("languages").each do |language|
          metrics = acceptance.fetch("by_language").fetch(language)
          failures = ReleasePolicy.failures(metrics, expected_labels: (0...18).to_a)
          passed &&= failures.empty?
          raise ArgumentError, "Release gates failed for #{language}" unless preview || failures.empty?
        end
        raise ArgumentError, "Inconsistent acceptance result" unless acceptance.fetch("passed") == passed
      end
      private_class_method :validate_acceptance!

      def self.load(path, device: "auto")
        path = File.expand_path(path)
        hashes = JSON.parse(File.read(File.join(path, "bundle-manifest.json")))
        %w[release.json acceptance.json runtime.json checkpoint/manifest.json checkpoint/metadata.json checkpoint/weights.pt checkpoint/tokenizer.json].each do |name|
          hashes.fetch(name)
        end
        hashes.each do |name, digest|
          unless %w[release.json acceptance.json runtime.json].include?(name) ||
              (name.start_with?("checkpoint/") && name.split("/").size == 2 && name.split("/").last != "..")
            raise ArgumentError, "Invalid release filename"
          end
          raise ArgumentError, "Release checksum mismatch: #{name}" unless Digest::SHA256.file(File.join(path, name)).hexdigest == digest
        end
        metadata = JSON.parse(File.read(File.join(path, "release.json")))
        raise ArgumentError, "Unsupported release" unless metadata.fetch("version") == "0.1" && metadata.fetch("profile") == "bilingual-request-domains"
        new(predictor: Predictor.load(File.join(path, "checkpoint"), device: device,
          candidate_chunk_size: metadata.fetch("candidate_chunk_size")), metadata: metadata)
      end

      def initialize(predictor:, metadata:)
        @predictor, @metadata = predictor, metadata
      end

      def route(state:, language:)
        probabilities(state: state, question: question(language), options: options(language), language: language)
      end

      def options(language)
        index = locale_index(language)
        Data::NaturalAdapter::SCENARIOS.map { |id, texts| { id: id, text: texts.fetch(index) } }
      end

      def probabilities(state:, question:, options:, language: nil)
        result = predictor.probabilities(state: state, question: question, options: options)
        texts = options.map { |option| option[:text] || option["text"] }.sort
        supported = metadata.fetch("languages").key?(language) && question == self.question(language) &&
          texts == self.options(language).map { |option| option.fetch(:text) }.sort && result.fetch("truncated_segments").zero?
        policy = metadata.fetch("languages")[language]
        confidence = result.fetch("probabilities").values.max
        eligible = supported && policy.fetch("available") && confidence >= policy.fetch("threshold")
        result.merge("release" => metadata.fetch("version"), "release_status" => metadata.fetch("status", "preview"),
          "supported_profile" => supported,
          "review_required" => !eligible, "suggested_option" => result.fetch("probabilities").max_by { |_id, value| value }.first)
      end

      private

      def question(language)
        locale_index(language).zero? ? "这项请求属于哪个领域？" : "Which domain does this request belong to?"
      end

      def locale_index(language)
        raise ArgumentError, "Supported routing languages: zh-CN, en-US" unless %w[zh-CN en-US].include?(language)
        language == "zh-CN" ? 0 : 1
      end
    end
  end
end
