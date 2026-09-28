require "faraday"
require "uri"

module EasyAI
  module Distillation
    module Teachers
      # Chat content only. Written confidence values are never promoted to logits.
      class LocalHttp
        attr_reader :signature

        def initialize(endpoint:, model:, revision:, backend_revision:, parameters: {}, connection: nil)
          uri = URI(endpoint)
          unless uri.scheme == "http" && %w[127.0.0.1 localhost ::1 [::1]].include?(uri.host) && !uri.userinfo && !uri.query && !uri.fragment
            raise ArgumentError, "Teacher endpoint must be a loopback HTTP URL without credentials or query"
          end
          [model, revision, backend_revision].each do |value|
            raise ArgumentError, "Teacher model and revisions must be explicit" unless value.is_a?(String) && !value.strip.empty?
          end
          allowed = %w[temperature top_p top_k min_p seed max_tokens chat_template_kwargs]
          raise ArgumentError, "Unsupported teacher parameters" unless parameters.is_a?(Hash) && (parameters.keys - allowed).empty?
          @signature = { "backend" => "local_chat_v1", "endpoint" => endpoint, "model" => model,
            "revision" => revision, "backend_revision" => backend_revision, "capabilities" => ["text"],
            "parameters" => { "temperature" => 0, "seed" => 1337, "max_tokens" => 64 }.merge(parameters) }
          @connection = connection || Faraday.new(url: endpoint, proxy: nil) do |client|
            client.options.open_timeout = 10
            client.options.read_timeout = 120
            client.adapter :net_http
          end
        end

        def call(request)
          body = signature.fetch("parameters").merge(request).merge("model" => signature.fetch("model"), "stream" => false)
          attempts = 0
          begin
            attempts += 1
            response = @connection.post("#{URI(signature.fetch('endpoint')).path.sub(%r{/$}, '')}/chat/completions",
              JSON.generate(body), "Content-Type" => "application/json")
            if response.status == 429 || response.status >= 500
              raise Faraday::ConnectionFailed, "Teacher temporarily unavailable (HTTP #{response.status})"
            end
            raise ArgumentError, "Teacher HTTP #{response.status}" unless response.success?
          rescue Faraday::ConnectionFailed, Faraday::TimeoutError
            raise if attempts >= 3
            sleep(0.2 * attempts)
            retry
          end
          result = JSON.parse(response.body)
          choice = result.fetch("choices").first
          raise ArgumentError, "Teacher response did not finish normally" unless choice && choice["finish_reason"] == "stop"
          content = choice.fetch("message").fetch("content")
          raise ArgumentError, "Teacher returned empty content" unless content.is_a?(String) && !content.strip.empty?
          { "kind" => "text", "text" => content, "usage" => result["usage"] }
        end
      end
    end
  end
end
