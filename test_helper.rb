require "minitest/autorun"
require "minitest/reporters"
Minitest::Reporters.use!
Minitest.load :minitest_reporter

require "debug"

ENV["LOG_PATH"] = File.expand_path("log/test.log", __dir__)
# ENV["LOG_LEVEL"] = "INFO"

# $LOAD_PATH.unshift File.expand_path("./", __dir__)
