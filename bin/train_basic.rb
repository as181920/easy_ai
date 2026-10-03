#!/usr/bin/env ruby
# Compatibility entry point for the historical teaching demo.
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
warn "Teaching demo moved to learning/12_gpt/train.rb"
load File.expand_path("../learning/12_gpt/train.rb", __dir__)
