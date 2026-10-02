require "test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/robust"

class RobustTest < Minitest::Test
  def generated(identity = 1, language = "en-US", split: "train")
    EasyAI::Decision::Data::RobustFacts.new("transport", identity, language, split: split).rows
  end

  def test_gold_follows_world_facts_and_unknown_is_not_false
    rows = %w[en-US zh-CN].flat_map { |language| (0...4).flat_map { |identity| generated(identity, language) } }
    rows.each do |row|
      world = row.fetch("world")
      facts, actor = world.values_at("facts", "queried_actor")
      expected = facts.key?(actor) ? (facts.fetch(actor) == world.fetch("assertion") ? "yes" : "no") : "unknown"

      assert_equal expected, row.fetch("target")
      EasyAI::Decision::Data::Example.new(row)
    end

    assert_equal %w[no unknown yes], rows.map { |row| row.fetch("target") }.uniq.sort
  end

  def test_same_state_actor_questions_and_order_are_independently_varied
    rows = generated.select { |row| row.fetch("world").fetch("variant") == "base" && row.fetch("target") != "unknown" }
    states = rows.group_by { |row| row.fetch("state") }

    assert_equal 2, states.size
    assert states.values.all? { |items| items.size == 4 && items.map { |row| row.fetch("world").fetch("queried_actor") }.uniq.size == 2 }
    assert_equal [0, 1], rows.map { |row| row.fetch("world").fetch("queried_position") }.uniq.sort
    assert_equal 4, (0...4).map { |identity| generated(identity).first.fetch("world").fetch("truth_pattern") }.uniq.size
  end

  def test_irrelevant_fact_truth_is_not_determined_by_the_first_actor
    %w[en-US zh-CN].each do |language|
      worlds = (0...32).map { |identity| generated(identity, language).find { |row| row.dig("world", "variant") == "irrelevant" }.fetch("world") }
      combinations = worlds.map { |world| [world.fetch("facts").values.first, world.fetch("facts").values.last] }.uniq

      assert_equal 4, combinations.size
    end
  end

  def test_expression_candidate_families_and_semantic_groups_are_held_out
    train = generated
    development = generated(1, "en-US", split: "validation")
    test = generated(1, "en-US", split: "test")

    assert_empty train.map { |row| row.fetch("group_id") } & test.map { |row| row.fetch("group_id") }
    assert_empty train.map { |row| row.fetch("world").fetch("candidate_wording") } & development.map { |row| row.fetch("world").fetch("candidate_wording") }
    assert_empty development.map { |row| row.fetch("world").fetch("candidate_wording") } & test.map { |row| row.fetch("world").fetch("candidate_wording") }
    assert_equal [2, 3], train.map { |row| row.fetch("options").size }.uniq.sort
  end

  def test_perfect_predictions_pass_every_contrast_and_constant_answers_do_not
    rows = %w[en-US zh-CN].flat_map { |language| generated(1, language) }
    correct = rows.map { |row| row.merge("prediction" => row.fetch("target")) }
    metrics = DecisionRobust.robust_metrics(correct)

    assert metrics.fetch("contrast_axes").values.all? { |cells| cells.values.all? { |cell| cell.fetch("all_correct") == 1.0 } }
    assert metrics.fetch("robust_by_language").values.all? { |cell| cell.fetch("world_all_correct") == 1.0 }
    constant = DecisionRobust.robust_metrics(rows.map { |row| row.merge("prediction" => "yes") })

    assert_in_delta 0.0, constant.fetch("robust_by_language").fetch("en-US").fetch("binding_all_correct")
    assert_in_delta 0.0, constant.fetch("robust_by_language").fetch("zh-CN").fetch("unknown_accuracy")
  end

  def test_actor_blind_predictions_cannot_pass_on_same_truth_worlds_alone
    rows = %w[en-US zh-CN].flat_map { |language| (0...4).flat_map { |identity| generated(identity, language) } }
    predictions = rows.map do |row|
      world = row.fetch("world")
      answer = row.fetch("target") == "unknown" ? "unknown" : world.fetch("facts").values.first == world.fetch("assertion") ? "yes" : "no"
      row.merge("prediction" => answer)
    end
    metrics = DecisionRobust.robust_metrics(predictions)

    assert_in_delta 0.5, metrics.fetch("robust_by_language").fetch("en-US").fetch("binding_all_correct")
    assert_in_delta 0.0, metrics.fetch("robust_by_language").fetch("en-US").fetch("mixed_binding_all_correct")
    assert_in_delta 1.0, metrics.fetch("robust_by_language").fetch("zh-CN").fetch("same_binding_all_correct")
    metrics["routing_by_language"] = %w[en-US zh-CN].to_h { |language| [language, { "accuracy" => 1.0 }] }
    gates = { "routing_tolerance" => 0.03, "minimum_known_accuracy" => 0.6, "minimum_unknown_accuracy" => 0.5,
      "minimum_binding_group_accuracy" => 0.4, "minimum_mixed_binding_group_accuracy" => 0.4, "minimum_wording_accuracy" => 0.5 }

    refute DecisionRobust.eligible?(metrics, metrics, gates)
  end

  def test_position_references_respect_candidate_counts_source_macro_and_ignore_routing
    Dir.mktmpdir do |directory|
      FileUtils.mkdir_p(File.join(directory, "data"))
      rows = %w[en-US zh-CN].flat_map { |language| generated(1, language) }
      rows << rows.first.merge("source" => "MASSIVE-Scenario", "target" => "unknown")
      path = File.join(directory, "data/test.jsonl")
      DecisionRobust.write_rows(path, rows)
      result = DecisionRobust.position_references(directory)

      assert_equal DecisionRobust.sha(path), result.fetch("dataset_sha256")
      { "first" => 0.25, "last" => 0.55, "uniform" => 0.35 }.each do |kind, expected|
        assert_in_delta expected, result.fetch("values").fetch(kind).fetch("factual_macro_accuracy"), 1e-12
      end
      assert_equal %w[en-US zh-CN], result.fetch("values").fetch("first").fetch("by_language").keys.sort
    end
  end

  def test_sampler_has_exact_declared_exposure_and_valid_pairs
    rows = %w[en-US zh-CN].flat_map do |language|
      raw = generated(1, language)
      raw + [raw.first.merge("id" => "public:#{language}", "source" => "BoolQ"), raw.first.merge("id" => "route:#{language}", "source" => "MASSIVE-Scenario")]
    end.map { |row| EasyAI::Decision::Data::Example.new(row) }
    sampler = EasyAI::Decision::Data::RobustSampler.new(rows)
    totals = Hash.new(0)
    5.times do |step|
      indexes = sampler.sample(32, rng: Random.new(step), step: step)

      assert_equal indexes, sampler.sample(32, rng: Random.new(step), step: step)
      indexes.each { |i| totals[rows[i].source] += 1 }
    end

    assert_equal [48, 48, 32, 32], totals.values_at("BoolQ", "Factual-V03-Known", "Factual-V03-Unknown", "MASSIVE-Scenario")
    invalid = rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("target" => row.target == "no" ? "yes" : row.target)) }

    assert_raises(ArgumentError) { EasyAI::Decision::Data::RobustSampler.new(invalid) }
  end

  def test_sampler_rejects_incomplete_groups_before_training
    rows = %w[en-US zh-CN].flat_map do |language|
      raw = generated(1, language)
      raw + [raw.first.merge("id" => "public:#{language}", "source" => "BoolQ"), raw.first.merge("id" => "route:#{language}", "source" => "MASSIVE-Scenario")]
    end.map { |row| EasyAI::Decision::Data::Example.new(row) }
    rows.delete_at(0)

    assert_raises(ArgumentError) { EasyAI::Decision::Data::RobustSampler.new(rows) }
    malformed = generated.map { |row| row.merge("prediction" => row.fetch("target")) }.drop(1)

    assert_raises(RuntimeError) { DecisionRobust.robust_metrics(malformed) }
  end

  def test_gradient_accumulation_preserves_the_effective_ce_update_with_mixed_candidate_counts
    Dir.mktmpdir do |directory|
      raw = %w[en-US zh-CN].flat_map do |language|
        rows = generated(1, language)
        rows + [rows.first.merge("id" => "public:#{language}", "source" => "BoolQ"), rows.first.merge("id" => "route:#{language}", "source" => "MASSIVE-Scenario")]
      end
      dataset = write_dataset(File.join(directory, "data.jsonl"), raw)
      cfg = tiny_config(model: { score_mode: "matching", embedding_norm: true, position_scale: 0.02 },
        input: { state_max_tokens: 256, question_option_max_tokens: 256 },
        training: { choice_microbatch: 4, gradient_accumulation: 8, track_coverage: true, steps: 1 })
      Torch.manual_seed(1337)
      model = EasyAI::Decision::ChoiceModel.new(cfg)
      other = EasyAI::Decision::ChoiceModel.new(cfg.with(training: { choice_microbatch: 8, gradient_accumulation: 4 }))
      other.load_state_dict(model.state_dict)
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      first = EasyAI::Decision::RobustTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "micro4"))
      second = EasyAI::Decision::RobustTrainer.new(model: other, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "micro8"))
      before = model.encoder.embedding.weight.detach.clone
      first.train
      second.train

      assert_operator (model.encoder.embedding.weight - before).abs.max.item, :>, 0
      assert_in_delta first.state.fetch("last_train_loss"), second.state.fetch("last_train_loss"), 1e-5
      assert_equal first.state.fetch("coverage"), second.state.fetch("coverage")
      assert first.model.named_parameters.all? { |name, tensor| (tensor - second.model.named_parameters.fetch(name)).abs.max.item < 1e-5 }
    end
  end

  def test_split_audit_rejects_shared_worlds_and_material_under_different_ids
    row = generated.first
    collator = mock("collator")
    panels = { "control-train.jsonl" => [row], "candidate-train.jsonl" => [],
      "validation.jsonl" => [row], "calibration.jsonl" => [], "test.jsonl" => [] }

    error = assert_raises(RuntimeError) { DecisionRobust.audit_splits!(panels, Set.new, collator) }

    assert_match(/Semantic group leakage/, error.message)
    panels["validation.jsonl"] = [row.merge("id" => "different", "group_id" => "different-world")]
    error = assert_raises(RuntimeError) { DecisionRobust.audit_splits!(panels, Set.new, collator) }

    assert_match(/State material leakage/, error.message)
  end

  def test_resume_preserves_ce_updates_coverage_and_sampler_configuration
    Dir.mktmpdir do |directory|
      cfg = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 },
        training: { choice_microbatch: 4, gradient_accumulation: 8, track_coverage: true, steps: 2 })
      raw = %w[en-US zh-CN].flat_map do |language|
        rows = generated(1, language)
        rows + [rows.first.merge("id" => "public:#{language}", "source" => "BoolQ"), rows.first.merge("id" => "route:#{language}", "source" => "MASSIVE-Scenario")]
      end
      dataset = write_dataset(File.join(directory, "data.jsonl"), raw)
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      Torch.manual_seed(1337)
      model = EasyAI::Decision::ChoiceModel.new(cfg)
      other = EasyAI::Decision::ChoiceModel.new(cfg)
      other.load_state_dict(model.state_dict)
      uninterrupted = EasyAI::Decision::RobustTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "full"))
      partial = EasyAI::Decision::RobustTrainer.new(model: other, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "partial"))
      uninterrupted.train
      partial.train(steps: 1)
      resumed = EasyAI::Decision::RobustTrainer.resume(partial.last_checkpoint, dataset: dataset, output: File.join(directory, "resumed"))
      resumed.train

      assert_tensor_close uninterrupted.model.encoder.embedding.weight, resumed.model.encoder.embedding.weight
      assert_equal uninterrupted.state.fetch("coverage"), resumed.state.fetch("coverage")
      assert_equal 64, resumed.state.fetch("examples_seen")
      restored = EasyAI::Decision::Checkpoint.load(resumed.last_checkpoint)
      restored.fetch(:metadata).fetch("training")["robust_sampling"] = { "version" => 99 }

      assert_raises(ArgumentError) do
        EasyAI::Decision::RobustTrainer.new(model: restored.fetch(:model), tokenizer: tokenizer, dataset: dataset,
          output: File.join(directory, "changed"), restored: restored.fetch(:metadata))
      end
    end
  end

  def test_incomplete_pilots_cannot_open_acceptance
    Dir.mktmpdir do |directory|
      DecisionRobust.stubs(:verify).returns({})
      File.write(File.join(directory, "confirmation.json"), JSON.generate("confirmation_required" => false))
      FileUtils.mkdir_p(File.join(directory, "control-1337"))
      File.write(File.join(directory, "control-1337/summary.json"), JSON.generate("step" => 10))

      error = assert_raises(RuntimeError) { DecisionRobust.final_candidates(directory) }

      assert_match(/incomplete/, error.message)
      refute_path_exists File.join(directory, "acceptance-opened.json")
    end
  end

  def test_opened_acceptance_is_not_reused_and_frozen_policy_changes_are_rejected
    Dir.mktmpdir do |directory|
      DecisionRobust.stubs(:final_candidates).returns({})
      File.write(File.join(directory, "acceptance-opened.json"), "{}")

      error = assert_raises(RuntimeError) { DecisionRobust.evaluate(directory) }

      assert_match(/already opened/, error.message)
      DecisionRobust.stubs(:verify).returns({})
      File.write(File.join(directory, "protocol.json"), "original")
      File.write(File.join(directory, "acceptance-opened.json"), JSON.generate("protocol_sha256" => "not-original"))

      assert_raises(RuntimeError) { DecisionRobust.verify_opened(directory) }
    end
  end

  def test_control_selection_audit_does_not_promote_previously_ineligible_points
    Dir.mktmpdir do |directory|
      root = File.join(directory, "control-1337")
      FileUtils.mkdir_p(root)
      FileUtils.mkdir_p(File.join(directory, "data"))
      File.write(File.join(directory, "data/validation.jsonl"), "")
      File.write(File.join(directory, "baseline.json"), JSON.generate("validation" => {}))
      File.write(File.join(root, "summary.json"), JSON.generate("selected" => nil, "step" => 2000))
      File.write(File.join(root, "validation.json"), JSON.generate([{ "step" => 100, "eligible" => false }]))
      DecisionRobust.stubs(:verify).returns({})
      DecisionRobust.expects(:measure).never
      DecisionRobust.audit_control_selection(directory)
      result = JSON.parse(File.read(File.join(root, "selection-audit.json")))

      assert_nil result.fetch("selected")
      assert_empty result.fetch("rechecked")
    end
  end

  def test_control_selection_audit_removes_a_checkpoint_that_fails_mixed_binding
    Dir.mktmpdir do |directory|
      root = File.join(directory, "control-1337")
      checkpoint = File.join(root, "choice/checkpoints/step-00000100-a")
      FileUtils.mkdir_p(checkpoint)
      FileUtils.mkdir_p(File.join(directory, "data"))
      File.write(File.join(checkpoint, "metadata.json"), "{}")
      File.write(File.join(directory, "data/validation.jsonl"), "")
      training = File.join(directory, "data/control-train.jsonl")
      File.write(training, "unchanged")
      old = { "key" => [0.8, -0.2], "checkpoint" => checkpoint }
      File.write(File.join(root, "summary.json"), JSON.generate("selected" => old, "step" => 2000))
      File.write(File.join(root, "validation.json"), JSON.generate([{ "step" => 100, "eligible" => true }]))
      File.write(File.join(directory, "baseline.json"), JSON.generate("validation" => {}))
      DecisionRobust.stubs(:verify).returns({ "selection" => {} })
      DecisionRobust.stubs(:device).returns("cpu")
      model = mock("model")
      model.expects(:to).with("cpu")
      EasyAI::Decision::Checkpoint.expects(:load).with(checkpoint).returns(model: model, tokenizer: nil, path: checkpoint)
      DecisionRobust.expects(:measure).returns({ "factual_macro_accuracy" => 0.8, "factual_nll" => 0.2 })
      DecisionRobust.expects(:eligible?).returns(false)
      DecisionRobust.audit_control_selection(directory)
      summary = JSON.parse(File.read(File.join(root, "summary.json")))

      assert_nil summary.fetch("selected")
      assert_equal old, summary.fetch("original_selected")
      assert_equal 2000, summary.fetch("step")
      assert_equal "unchanged", File.read(training)
    end
  end
end
