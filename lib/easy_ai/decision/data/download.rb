require "faraday"
require "uri"
require "fileutils"
require "digest"

module EasyAI
  module Decision
    module Data
      class Download
        def self.massive(path, expected_sha256: nil)
          fetch(Adapters::Massive::URL, path, expected_sha256: expected_sha256)
        end

        def self.fetch(url, path, expected_sha256: nil, max_mib: 200, redirects: 5)
          raise ArgumentError, "File already exists: #{path}" if File.exist?(path)
          FileUtils.mkdir_p(File.dirname(path))
          uri = URI(url)
          raise ArgumentError, "Only HTTPS downloads are accepted" unless uri.scheme == "https"
          temporary = "#{path}.part"
          written = 0
          connection = Faraday.new do |client|
            client.options.open_timeout = 20
            client.options.read_timeout = 60
            client.adapter :net_http
          end
          response = File.open(temporary, "wb") do |file|
            connection.get(url) do |request|
              request.options.on_data = proc do |chunk, _total, _env|
                written += chunk.bytesize
                raise "Dataset exceeds #{max_mib} MiB download limit" if written > max_mib * 1024 * 1024
                file.write(chunk)
              end
            end
          end
          if (300...400).cover?(response.status)
            raise ArgumentError, "Too many download redirects" if redirects <= 0
            return fetch(URI.join(url, response.headers.fetch("location")).to_s, path,
              expected_sha256: expected_sha256, max_mib: max_mib, redirects: redirects - 1)
          end
          raise "Download failed: HTTP #{response.status}" unless response.success?
          sha = Digest::SHA256.file(temporary).hexdigest
          raise "Dataset checksum mismatch" if expected_sha256 && sha != expected_sha256
          File.rename(temporary, path)
          { "path" => path, "sha256" => sha, "bytes" => written, "source" => uri.to_s }
        ensure
          FileUtils.rm_f(temporary) if temporary
        end
      end
    end
  end
end
