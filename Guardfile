guard :bundler do
  require "guard/bundler"
  require "guard/bundler/verify"
  helper = Guard::Bundler::Verify.new

  files = ["Gemfile"]
  files += Dir["*.gemspec"] if files.any? { |f| helper.uses_gemspec?(f) }

  # Assume files are symlinked from somewhere
  files.each { |file| watch(helper.real_path(file)) }
end

guard :rubocop, cli: ["--parallel", "--format", "fuubar"], cmd: "bin/rubocop" do
  watch(/.+\.rb$/)
  watch(%r{^app/views/(.+)/.+})
  watch(%r{(?:.+/)?\.rubocop(?:_todo)?\.yml$}) { |m| File.dirname(m[0]) }
end

require "debug"
guard :minitest do
  # with Minitest::Unit
  watch(%r{\A(.+)\.rb\z}) { |m| "test/#{m[1]}_test.rb" }
  watch(%r{^test/(.*)/?(.*)_test\.rb$})
  watch(%r{\Atest_helper\.rb$}) { "test" }
end
