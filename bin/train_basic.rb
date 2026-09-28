#!/usr/bin/env ruby
# Compatibility entry point for the historical teaching demo.
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
warn "Teaching demo moved to learning/transformer/train.rb"
load File.expand_path("../learning/transformer/train.rb", __dir__)
