#!/usr/bin/env ruby
require "bundler/setup"
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"
EasyAILearning::Course::Cli.run("16_capstone", EasyAILearning::Capstone::Experiment)
