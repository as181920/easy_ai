#!/usr/bin/env ruby
# Explicit historical text-training options select the corpus trainer.
if ARGV.any? { |arg| %w[--data -d --tokenizer -t --iters -i --prompt -p].include?(arg) }
  load File.expand_path("train_text.rb", __dir__)
else
  require "bundler/setup"
  $LOAD_PATH.unshift File.expand_path("../lib", __dir__)
  require "easy_ai_learning"
  EasyAILearning::Course::Cli.run("12_gpt", EasyAILearning::GPT::Experiment)
end
