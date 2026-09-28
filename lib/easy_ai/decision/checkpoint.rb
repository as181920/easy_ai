require "json"
require "fileutils"
require "digest"
require "securerandom"
require "time"

module EasyAI
  module Decision
    class Checkpoint
      FORMAT_VERSION = 1

      def self.save(root, model:, tokenizer:, optimizer: nil, training_state: {}, calibration: nil)
        FileUtils.mkdir_p(File.join(root, "checkpoints"))
        name = "step-#{format('%08d', training_state.fetch('step', 0))}-#{SecureRandom.hex(4)}"
        staging = File.join(root, "checkpoints", ".#{name}.tmp")
        destination = File.join(root, "checkpoints", name)
        FileUtils.mkdir_p(staging)
        weights = model.state_dict.to_h { |key, tensor| [key, tensor.detach.cpu.clone] }
        Torch.save(weights, File.join(staging, "weights.pt"))
        optimizer_state = optimizer&.state_dict
        if optimizer_state
          tensors = optimizer_state.delete("tensors")
          Torch.save(tensors, File.join(staging, "optimizer.pt")) unless tensors.empty?
        end
        tokenizer.save(File.join(staging, "tokenizer.json"))
        metadata = { "format_version" => FORMAT_VERSION, "created_at" => Time.now.utc.iso8601,
                    "config" => model.config.to_h, "training" => training_state,
                    "optimizer" => optimizer_state, "calibration" => calibration,
                    "tokenizer_fingerprint" => tokenizer.fingerprint,
                    "torch_rb_version" => Gem.loaded_specs.fetch("torch-rb").version.to_s }
        File.write(File.join(staging, "metadata.json"), JSON.pretty_generate(metadata))
        manifest = Dir.children(staging).sort.to_h { |file| [file, Digest::SHA256.file(File.join(staging, file)).hexdigest] }
        File.write(File.join(staging, "manifest.json"), JSON.pretty_generate(manifest))
        File.rename(staging, destination)
        atomic_json(File.join(root, "latest.json"), { "checkpoint" => "checkpoints/#{name}" })
        destination
      ensure
        FileUtils.rm_rf(staging) if staging && File.directory?(staging)
      end

      def self.resolve(path)
        path = File.expand_path(path)
        return path if File.file?(File.join(path, "manifest.json"))
        relative = JSON.parse(File.read(File.join(path, "latest.json"))).fetch("checkpoint")
        destination = File.expand_path(relative, path)
        raise ArgumentError, "Checkpoint pointer escapes run directory" unless destination.start_with?(path + File::SEPARATOR)
        destination
      end

      def self.load(path)
        path = resolve(path)
        manifest = JSON.parse(File.read(File.join(path, "manifest.json")))
        %w[weights.pt tokenizer.json metadata.json].each { |name| manifest.fetch(name) }
        manifest.each do |file, checksum|
          raise ArgumentError, "Invalid checkpoint filename" unless file == File.basename(file)
          raise ArgumentError, "Checkpoint checksum mismatch: #{file}" unless Digest::SHA256.file(File.join(path, file)).hexdigest == checksum
        end
        metadata = JSON.parse(File.read(File.join(path, "metadata.json")))
        raise ArgumentError, "Unsupported checkpoint version" unless metadata["format_version"] == FORMAT_VERSION
        config = Config.new(metadata.fetch("config"))
        tokenizer = Tokenizers::Registry.load(File.join(path, "tokenizer.json"))
        raise ArgumentError, "Tokenizer mismatch" unless tokenizer.fingerprint == metadata.fetch("tokenizer_fingerprint")
        raise ArgumentError, "Tokenizer exceeds model vocabulary" if tokenizer.vocab_size > config[:model]["vocab_size"]
        model = ChoiceModel.new(config)
        model.load_state_dict(Torch.load(File.join(path, "weights.pt")))
        if metadata["optimizer"]
          metadata["optimizer"]["tensors"] = manifest.key?("optimizer.pt") ? Torch.load(File.join(path, "optimizer.pt")) : {}
        end
        { model: model, tokenizer: tokenizer, metadata: metadata, path: path,
         weights_fingerprint: manifest.fetch("weights.pt") }
      end

      def self.atomic_json(path, value)
        temporary = "#{path}.#{SecureRandom.hex(4)}.tmp"
        File.write(temporary, JSON.pretty_generate(value))
        File.rename(temporary, path)
      ensure
        FileUtils.rm_f(temporary) if temporary
      end
    end
  end
end
