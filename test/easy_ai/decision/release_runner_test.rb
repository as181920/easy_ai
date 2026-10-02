require "test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/release"

class ReleaseRunnerTest < Minitest::Test
  def test_memory_failure_replays_the_same_fixed_evaluation_on_cpu
    Dir.mktmpdir do |root|
      FileUtils.mkdir_p(File.join(root, "data"))
      write_dataset(File.join(root, "data/validation.jsonl"), [example])
      model = EasyAI::Decision::ChoiceModel.new(tiny_config)
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      constructor = SemanticCoverageEvaluation.method(:new)
      failed = Object.new
      failed.define_singleton_method(:collect) { |_rows| raise EasyAI::Runtime::DevicePolicy::MemoryBudgetExceeded, "simulated budget" }
      data = logits = nil
      SemanticCoverageEvaluation.stubs(:new).with(has_entries(device: "cuda")).returns(failed)
      SemanticCoverageEvaluation.stubs(:new).with(has_entries(device: "cpu")).returns(constructor.call(model: model, tokenizer: tokenizer, device: "cpu", batch_size: 4))
      data, logits = DecisionRelease.collect(root, "validation", { model: model, tokenizer: tokenizer }, "cuda")

      assert_equal 1, data.size
      assert logits.flatten.all?(&:finite?)
      assert_equal "cpu", model.parameters.first.device.type.to_s
    end
  end

  def test_acceptance_cannot_be_reopened
    Dir.mktmpdir do |root|
      File.write(File.join(root, "protocol.json"), JSON.generate("requirements" => EasyAI::Decision::ReleasePolicy::REQUIREMENTS, "files_sha256" => {}))
      File.write(File.join(root, "evaluation-unit.json"), JSON.generate("evaluation_unit" => DecisionRelease::EVALUATION_UNIT))
      File.write(File.join(root, "acceptance-opened.json"), "{}")
      error = assert_raises(RuntimeError) { DecisionRelease.acceptance(root) }

      assert_match(/Acceptance already opened/, error.message)
    end
  end
end
