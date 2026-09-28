module EasyAI
  module Tokenizers
    class Registry
      def self.build(backend)
        case backend.to_s
        when "ruby" then ByteBpe.new
        when "native" then NativeBpe.new
        else raise ArgumentError, "Tokenizer backend must be ruby or native"
        end
      end

      def self.load(path)
        data = JSON.parse(File.read(path))
        case data.fetch("format")
        when "easy_ai.byte_bpe" then ByteBpe.from_h(data)
        when "easy_ai.native_bpe" then NativeBpe.from_h(data)
        else raise ArgumentError, "Unknown tokenizer format"
        end
      end
    end
  end
end
