#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "json"
require_relative "../../lib/easy_ai"

checkpoint = ARGV.fetch(0) { abort "Usage: ruby benchmarks/decision/semantic_probes.rb CHECKPOINT" }
predictor = EasyAI::Decision::Predictor.load(checkpoint, device: "cpu")
probes = [
  { task: "untrained_negation", state: "来不及了要迟到了", question: "会迟到么", options: [{ id: 0, text: "会" }, { id: 1, text: "不会" }], expected: "0" },
  { task: "untrained_negation", state: "来得及不会迟到了", question: "会迟到么", options: [{ id: 0, text: "会" }, { id: 1, text: "不会" }], expected: "1" },
  { task: "untrained_negation", state: "不会迟到了", question: "会迟到么", options: [{ id: 0, text: "会" }, { id: 1, text: "不会" }], expected: "1" },
  { task: "untrained_negation", state: "我要迟到了", question: "会迟到么", options: [{ id: 0, text: "会" }, { id: 1, text: "不会" }], expected: "0" },
  { task: "untrained_negation", state: "I think i will be late", question: "will I late?", options: [{ id: 0, text: "will" }, { id: 1, text: "no" }], expected: "0" },
  { task: "untrained_negation", state: "I will not be late", question: "Will I be late?", options: [{ id: 0, text: "yes" }, { id: 1, text: "no" }], expected: "1" },
  { task: "intent", state: "明天七点叫我起床", question: "Intent?", options: [{ id: "alarm_set", text: "alarm set" }, { id: "music_play", text: "music play" }], expected: "alarm_set" },
  { task: "intent", state: "播放音乐", question: "Intent?", options: [{ id: "alarm_set", text: "alarm set" }, { id: "music_play", text: "music play" }], expected: "music_play" }
]
results = probes.map do |probe|
  prediction = predictor.probabilities(**probe.slice(:state, :question, :options))
  probe.merge(result: prediction, chosen: prediction.fetch("probabilities").max_by { |_id, value| value }.first)
end
puts JSON.pretty_generate(checkpoint: checkpoint, note: "Manual probes, not a held-out benchmark. These probes are never deliberately added to training; public data was not searched for accidental overlap.", results: results)
