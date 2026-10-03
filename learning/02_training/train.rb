#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"
EasyAILearning::Course::Cli.run("02_training", EasyAILearning::Training::Experiment)
