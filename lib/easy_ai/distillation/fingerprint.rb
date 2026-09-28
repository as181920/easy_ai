require "json"
require "digest"

module EasyAI
  module Distillation
    module Fingerprint
      def self.call(value)
        Digest::SHA256.hexdigest(JSON.generate(canonical(value)))
      end

      def self.canonical(value)
        case value
        when Hash then value.keys.sort.to_h { |key| [key, canonical(value.fetch(key))] }
        when Array then value.map { |item| canonical(item) }
        else value
        end
      end
    end
  end
end
