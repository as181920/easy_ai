require "digest"
require "json"
require "fileutils"

module EasyAI
  module Decision
    module Data
      class SemanticSources
        FILES = {
          "ocnli-train.jsonl" => ["https://raw.githubusercontent.com/CLUEbenchmark/OCNLI/main/data/ocnli/train.50k.json", "cb47a6c00105bfb49bbb791dd537b284d3138f9f1ad5cd1e187afc470ff004e0", "CC-BY-NC-2.0"],
          "ocnli-dev.jsonl" => ["https://raw.githubusercontent.com/CLUEbenchmark/OCNLI/main/data/ocnli/dev.json", "001ded235d73a18dac68a45a38cd5f9b3f9311fa4246df623e199422c073ddc5", "CC-BY-NC-2.0"],
          "dureader-yesno.tar.gz" => ["https://bj.bcebos.com/paddlenlp/datasets/dureader_yesno-data.tar.gz", "c6fdbc76771e0eb3c36de78017da64a1097992536d204e98eb5d7e32abe1c44d", "LUGE participant agreement; noncommercial research; see archive License.pdf"],
          "boolq.zip" => ["https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip", "853fbe7922f70c59629f06a39e8d9ca440c3d740e760fd3b87a5ddf3dcba2436", "CC-BY-SA-3.0; BoolQ via SuperGLUE"]
        }.freeze

        def self.prepare(directory)
          FileUtils.mkdir_p(directory)
          records = FILES.map do |name, (url, sha, license)|
            path = File.join(directory, name)
            Download.fetch(url, path, expected_sha256: sha) unless File.file?(path)
            raise ArgumentError, "Source checksum mismatch: #{name}" unless Digest::SHA256.file(path).hexdigest == sha
            { "file" => name, "source" => url, "sha256" => sha, "license" => license }
          end
          File.write(File.join(directory, "sources.json"), JSON.pretty_generate(records))
          records
        end
      end
    end
  end
end
