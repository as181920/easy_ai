require_relative "../../lib/easy_ai"

module EasyAILearning
  def self.logger
    EasyAI.logger
  end

  Logger = EasyAI::Logger
end

loader = Zeitwerk::Loader.new
loader.tag = "easy_ai_learning"
loader.inflector.inflect("gpt" => "GPT", "basic_nn" => "BasicNN", "cnn" => "CNN", "rnn" => "RNN", "rl" => "RL", "k_means" => "KMeans")
loader.push_dir(File.join(__dir__, "easy_ai_learning"), namespace: EasyAILearning)
loader.setup

EasyAILearning.define_singleton_method(:loader) { loader }
