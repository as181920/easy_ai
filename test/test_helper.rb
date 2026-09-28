ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "minitest/autorun"
require "minitest/reporters"
Minitest.load :minitest_reporter
Minitest::Reporters.use!

require "tmpdir"
require "stringio"

ENV["LOG_PATH"] = File.expand_path("../log/test.log", __dir__)

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai"

module DecisionTestSupport
  def tiny_config(**overrides)
    EasyAI::Decision::Config.new({
      model: { vocab_size: 262, hidden_size: 16, encoder_layers: 1, attention_heads: 2, ffn_size: 32, dropout: 0.0 },
      input: { state_max_tokens: 64, question_option_max_tokens: 48 },
      training: { device: "cpu", choice_microbatch: 2, mlm_microbatch: 2, gradient_accumulation: 1,
                 steps: 4, warmup_steps: 0, eval_every: 2, checkpoint_every: 2 }
    }.deep_merge(overrides))
  end

  def example(id: "sample", state: "red", target: "r", language: "en", options: nil)
    EasyAI::Decision::Data::Example.new({ id: id, group_id: id, language: language,
      state: state, question: "Color?", options: options || [{ id: "r", text: "red" }, { id: "b", text: "blue" }], target: target })
  end

  def write_dataset(path, examples)
    File.write(path, examples.map { |e| JSON.generate(e.respond_to?(:to_h) ? e.to_h : e) }.join("\n") + "\n")
    EasyAI::Decision::Data::Dataset.new(path)
  end

  def assert_tensor_close(expected, actual, tolerance = 1e-6)
    assert_equal expected.shape, actual.shape
    assert_operator (expected - actual).abs.max.item, :<=, tolerance
  end
end

class Minitest::Test
  include DecisionTestSupport
end
