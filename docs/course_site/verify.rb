require "cgi"
require "json"
require "optparse"
require "uri"

root = File.expand_path("../../tmp/learning-site", __dir__)
OptionParser.new { |parser| parser.on("--directory PATH") { |path| root = File.expand_path(path) } }.parse!
abort "Build the site first: bin/learning-docs build" unless File.file?(File.join(root, "index.html"))
prefix = ENV.fetch("LEARNING_SITE_BASEURL", "").delete_suffix("/")
count = 0
Dir.glob("#{root}/**/*.html").each do |file|
  File.read(file).scan(/(?:href|src)=["']([^"']+)["']/).flatten.each do |raw|
    escaped = CGI.unescapeHTML(raw).gsub(/[^\x21-\x7e]/) { |char| URI::DEFAULT_PARSER.escape(char) }
    url = URI.parse(escaped)
    next if url.scheme || url.host || url.path.empty?
    path = URI::DEFAULT_PARSER.unescape(url.path)
    if path.start_with?("/") && !prefix.empty?
      abort "Link outside site prefix: #{raw}" unless path.start_with?(prefix + "/")
      path = path.delete_prefix(prefix)
    end
    target = path.start_with?("/") ? root + path : File.expand_path(path, File.dirname(file))
    target += "/index.html" if File.directory?(target)
    abort "Broken link: #{file.delete_prefix(root)} -> #{raw}" unless File.file?(target)
    count += 1
  end
end
abort "Expected all 17 chapters" unless Dir.glob("#{root}/learning/[0-9][0-9]_*/README.html").size == 17
index = JSON.parse(File.read("#{root}/assets/js/search-data.json"))
abort "Missing searchable course content" unless index.values.any? { |row| row["content"].to_s.include?("AdamW") }
abort "Source pages must be excluded from search" if index.values.any? { |row| row["url"].to_s.end_with?(".rb.html") }
origin = ENV["LEARNING_SITE_URL"].to_s.delete_suffix("/")
unless origin.empty?
  home = File.read(File.join(root, "index.html"))
  expected = CGI.escapeHTML(origin + prefix + "/")
  abort "Incorrect production canonical URL" unless home.include?(%(<link rel="canonical" href="#{expected}"))
end
puts "Verified 17 chapters, #{count} local links/assets, search, and deployment URLs."
