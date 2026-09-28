require "active_support/all"
require "torch"
require "zeitwerk"

module EasyAI
  def self.logger
    EasyAI::Logger.logger
  end
end

loader = Zeitwerk::Loader.new
loader.tag = "easy_ai"
loader.inflector.inflect("nn" => "NN", "adamw" => "AdamW")
loader.push_dir(File.join(__dir__, "easy_ai"), namespace: EasyAI)
loader.setup

EasyAI.define_singleton_method(:loader) { loader }
