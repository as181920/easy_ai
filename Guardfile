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
  watch(%r{(?:.+/)?\.rubocop(?:_todo)?\.yml$}) { |m| File.dirname(m[0]) }
end

require "debug"
guard :minitest do
  watch(%r{\Alib/easy_ai/(decision|nn|optim|runtime)/.+\.rb\z}) { "test/easy_ai/decision" }
  watch(%r{\Alib/easy_ai/tokenizers/.+\.rb\z}) { "test/easy_ai/tokenizers" }
  watch(%r{\Alib/easy_ai\.rb\z}) { "test" }
  watch(%r{\Atest/.+_test\.rb\z})
  watch(%r{\Atest/test_helper\.rb\z}) { "test" }
end
