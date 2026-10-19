require_relative "../../test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/relational_encoding"

# Simulate the prediction boundary with explicit correct/wrong vectors; no
# convergence or trained-model quality is assumed by these workflow tests.
class RelationalAcceptanceTest < Minitest::Test
  def fixture(root, fail_acceptance: false)
    directory = File.join(root, "pilot")
    FileUtils.mkdir_p(File.join(directory, "data"))
    config = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 })
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    paths = %w[parent candidate confirmation].each_with_index.to_h do |name, index|
      model = EasyAI::Decision::ChoiceModel.new(config.with(training: { seed: index == 2 ? 2027 : 1337 }))
      Torch.no_grad { model.named_parameters.fetch("score.bias").fill!(index.to_f) }
      [name, EasyAI::Decision::Checkpoint.save(File.join(root, name), model: model, tokenizer: tokenizer, training_state: { "step" => 1 })]
    end
    old_parent = DecisionRelational::PARENT
    DecisionRelational.send(:remove_const, :PARENT)
    DecisionRelational.const_set(:PARENT, paths.fetch("parent"))
    @restore_parent = old_parent
    panels = EasyAI::Decision::Data::JudgmentCorpus.splits(counts: { "calibration" => 2, "test" => 2, "validation" => 2 })
      .transform_values { |rows| rows.select { |row| row.dig("world", "profile") == "binary" } }
    natural = panels.fetch("validation").map { |row| row.merge("profile" => "binary") }
    %w[calibration test].each { |name| DecisionRobust.write_rows(File.join(directory, "data/#{name}.jsonl"), panels.fetch(name)) }
    DecisionRobust.write_rows(File.join(directory, "data/natural-test.jsonl"), natural)
    good = panels.fetch("test").map { |row| row.fetch("options").map { |option| option.fetch("id") == row.fetch("target") ? 3.0 : 0.0 } }
    weak = good.map { [0.0, 0.0] }
    accurate = DecisionJudgment.metrics(panels.fetch("test"), good)
    baseline = DecisionJudgment.metrics(panels.fetch("test"), weak)
    requirements = { "binary_balanced" => 0.75, "mixed_groups" => 0.5, "parent_gain" => 0.05,
      "natural_minimum_accuracy" => 0.65, "natural_tolerance" => 0.03 }
    protocol = { "publication" => requirements, "files_sha256" => {} }
    DecisionRelational.write(File.join(directory, "protocol.json"), protocol)
    %w[1337 2027].zip(%w[candidate confirmation]).each do |seed, name|
      FileUtils.mkdir_p(File.join(directory, "seed-#{seed}"))
      DecisionRelational.write(File.join(directory, "seed-#{seed}/summary.json"), { "known_mastery_passed" => true,
        "selected" => { "binary" => { "checkpoint" => paths.fetch(name), "metrics" => { "controlled" => accurate, "natural" => accurate } } } })
    end
    DecisionRelational.write(File.join(directory, "selection.json"), { "profile" => "binary",
      "summary_sha256" => DecisionRelational.sha(File.join(directory, "seed-1337/summary.json")) })
    DecisionRelational.stubs(:pilot_protocol).returns(protocol)
    DecisionRelational.stubs(:pilot_baseline).returns({ "binary" => { "controlled" => baseline, "natural" => baseline } })
    DecisionRobust.stubs(:device).returns("cpu")
    result_buffer = []
    DecisionRobust.stubs(:collect).with do |model, _tok, raw, _device|
      correct = model.named_parameters.fetch("score.bias").item > 0.5
      correct &&= raw.all? { |row| row.fetch("group_id").include?("calibration") } if fail_acceptance
      examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
      logits = raw.map { |row| row.fetch("options").map { |option| correct && option.fetch("id") == row.fetch("target") ? 3.0 : 0.0 } }
      result_buffer.replace([examples, logits])
      true
    end.returns(result_buffer)
    directory
  end

  def teardown
    if @restore_parent
      DecisionRelational.send(:remove_const, :PARENT)
      DecisionRelational.const_set(:PARENT, @restore_parent)
    end
    super
  end

  def test_qualified_fixed_policy_calibrates_accepts_and_packages_a_loadable_preview
    Dir.mktmpdir do |root|
      directory = fixture(root)
      DecisionRelational.calibrate(root)
      result = DecisionRelational.accept(root)
      preview = DecisionRelational.package_preview(root)
      predictor = EasyAI::Decision::Predictor.load(preview, device: "cpu")
      value = predictor.probabilities(state: "record", question: "claim", options: [{ id: 1, text: "yes" }, { id: "no", text: "no" }])

      assert result.fetch("passed")
      assert_path_exists File.join(directory, "acceptance-opened.json")
      assert value.fetch("calibrated")
      assert_in_delta 1.0, value.fetch("probabilities").values.sum, 1e-6
      assert_raises(RuntimeError) { DecisionRelational.accept(root) }
    end
  end

  def test_fresh_acceptance_failure_prevents_packaging_despite_perfect_development
    Dir.mktmpdir do |root|
      directory = fixture(root, fail_acceptance: true)
      DecisionRelational.calibrate(root)
      result = DecisionRelational.accept(root)

      refute result.fetch("passed")
      assert_raises(RuntimeError) { DecisionRelational.package_preview(root) }
      refute_path_exists File.join(directory, "factual-preview")
    end
  end

  def test_calibration_cannot_change_even_before_acceptance_is_opened
    Dir.mktmpdir do |root|
      directory = fixture(root)
      DecisionRelational.calibrate(root)
      path = File.join(directory, "calibration.json")
      value = JSON.parse(File.read(path))
      value.fetch("policies").fetch("candidate")["temperature"] *= 2
      File.write(path, JSON.generate(value))

      assert_raises(RuntimeError) { DecisionRelational.frozen_acceptance(root, "standard") }
      refute_path_exists File.join(directory, "acceptance-opened.json")
    end
  end
end
