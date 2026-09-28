require "optparse"
require "yaml"
require "faraday"

module EasyAI
  module Distillation
    class Cli
      def self.run(argv, out: $stdout, err: $stderr, command: "distill-collect", teacher_factory: Teachers::LocalHttp.method(:new))
        options = {}
        parser = OptionParser.new do |flags|
          flags.banner = "Usage: bin/easy-ai #{command} --data FILE --output DIR [--teacher-config FILE --purpose train|development | --artifact DIR]"
          %i[data teacher_config output purpose artifact prompt_profile].each do |key|
            flags.on("--#{key.to_s.tr('_', '-')} VALUE") { |value| options[key] = value }
          end
          flags.on("--exclude FILES", Array, "Datasets whose groups must not overlap") { |value| options[:exclude] = value }
          flags.on("-h", "--help") { options[:help] = true }
        end
        parser.parse!(argv)
        if options[:help]
          out.puts(parser)
          return 0
        end
        raise ArgumentError, "Unexpected arguments" unless argv.empty?
        data = Decision::Data::Dataset.new(options.fetch(:data), require_target: false)
        exclusions = options.fetch(:exclude, []).map { |path| Decision::Data::Dataset.new(path) }
        exclusions.each { |dataset| Decision::Data::Dataset.assert_disjoint!(data, dataset) }
        if command == "distill-export"
          result = Decision::DistillationExport.write(dataset: data, artifact: options.fetch(:artifact), output: options.fetch(:output))
          out.puts(JSON.pretty_generate(result))
          return 0
        end
        config = YAML.safe_load_file(options.fetch(:teacher_config), aliases: false)
        teacher = teacher_factory.call(**config.transform_keys(&:to_sym))
        adapter = Decision::DistillationAdapter.new(profile: options.fetch(:prompt_profile, "generic"))
        artifact = Collector.new(teacher: teacher, adapter: adapter, output: options.fetch(:output),
          purpose: options.fetch(:purpose), progress: err).run(data)
        out.puts(JSON.pretty_generate(artifact.manifest.merge("artifact_sha256" => artifact.fingerprint)))
        0
      rescue ArgumentError, KeyError, OptionParser::ParseError, Errno::ENOENT, JSON::ParserError, Faraday::Error => error
        err.puts("easy-ai #{command}: #{error.message}")
        1
      end
    end
  end
end
