module EasyAI
  module Decision
    module DistillationExport
      # Turn teacher content into ordinary candidate training rows for unlabeled input.
      # Existing gold labels are never overwritten; mixed supervision uses the Trainer adapter.
      def self.write(dataset:, artifact:, output:)
        artifact = Distillation::Artifact.new(artifact) unless artifact.is_a?(Distillation::Artifact)
        adapter = DistillationAdapter.from_signature(artifact.manifest.fetch("adapter"))
        unless artifact.manifest["purpose"] == "train" && artifact.manifest["source_sha256"] == dataset.fingerprint &&
            artifact.manifest["adapter"] == adapter.signature && artifact.size == dataset.size
          raise ArgumentError, "Expected matching training teacher artifact"
        end
        raise ArgumentError, "Output already exists" if File.exist?(output)
        dataset.each do |row|
          raise ArgumentError, "Pseudo-label export requires unlabeled input; preserve existing gold labels" if row.target
          record = artifact.fetch(row.id)
          unless record["identity"] == adapter.identity(row) && record.dig("supervision", "kind") == "label"
            raise ArgumentError, "Expected aligned hard teacher label"
          end
          raise ArgumentError, "Teacher target not in options" unless row.options.any? { |option| option["id"] == record.dig("supervision", "target") }
        end
        FileUtils.mkdir_p(output)
        temporary = File.join(output, "train.jsonl.part")
        File.open(temporary, "w") do |file|
          dataset.each do |row|
            target = artifact.fetch(row.id).fetch("supervision").fetch("target")
            file.puts(JSON.generate(row.to_h.merge("target" => target,
              "label_provenance" => { "kind" => "teacher_pseudo_label", "artifact_sha256" => artifact.fingerprint })))
          end
        end
        path = File.join(output, "train.jsonl")
        File.rename(temporary, path)
        manifest = { "version" => 1, "kind" => "decision_pseudo_labels", "source_sha256" => dataset.fingerprint,
          "teacher_artifact_sha256" => artifact.fingerprint, "count" => dataset.size,
          "files_sha256" => { "train.jsonl" => Digest::SHA256.file(path).hexdigest } }
        Distillation::Artifact.write_json(File.join(output, "manifest.json"), manifest)
        manifest
      end
    end
  end
end
