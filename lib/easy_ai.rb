require "active_support/core_ext/hash/deep_merge"
require "active_support/core_ext/hash/keys"
require "active_support/core_ext/module/delegation"
require "active_support/core_ext/object/blank"
require "torch"
require "zeitwerk"

module EasyAI
end

loader = Zeitwerk::Loader.new
loader.tag = "easy_ai"
loader.inflector.inflect("gpt" => "GPT")
loader.push_dir(File.join(__dir__, "easy_ai"), namespace: EasyAI)
loader.setup

EasyAI.define_singleton_method(:loader) { loader }
