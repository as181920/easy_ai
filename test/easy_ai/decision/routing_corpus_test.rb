require "test_helper"

class RoutingCorpusTest < Minitest::Test
  def row(id:, state:, partition:, language: "en-US", target: "a")
    { "id" => "#{id}:#{language}", "group_id" => "official:#{id}", "source" => "MASSIVE-Scenario", "language" => language,
      "state" => state, "question" => "Domain?", "options" => [{ "id" => "a", "text" => "alarm" }, { "id" => "b", "text" => "music" }],
      "target" => target, "partition" => partition }
  end

  def split(rows, historical: [])
    material = EasyAI::Decision::Data::NaturalCorpus.method(:material)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: tiny_config)
    EasyAI::Decision::Data::RoutingCorpus.new(rows, historical_material: historical.map(&material).to_set,
      train_material: rows.map { |item| material.call(item.fetch("state")) }.to_set, collator: collator).splits
  end

  def test_one_observed_translation_excludes_the_entire_acceptance_group
    rows = [row(id: 1, state: "observed", partition: "test"), row(id: 1, state: "new", partition: "test", language: "zh-CN")]

    assert_empty split(rows, historical: ["observed"]).fetch("test")
  end

  def test_duplicate_material_cannot_cross_official_partitions
    rows = [row(id: 1, state: "same", partition: "train"), row(id: 2, state: "same", partition: "test")]
    result = split(rows)

    assert_empty result.fetch("train")
    assert_equal 1, result.fetch("test").size
  end

  def test_parallel_dev_rows_stay_together_and_never_train
    rows = [row(id: 3, state: "english", partition: "dev"), row(id: 3, state: "chinese", partition: "dev", language: "zh-CN")]
    result = split(rows)

    assert_empty result.fetch("train")
    assert_equal [0, 2], %w[validation calibration].map { |name| result.fetch(name).size }.sort
  end

  def test_conflicting_targets_are_excluded
    rows = [row(id: 1, state: "conflict", partition: "train"), row(id: 2, state: "conflict", partition: "train", target: "b")]

    assert_empty split(rows).fetch("train")
  end

  def test_statistics_use_one_target_independent_example_per_language_and_component
    examples = [row(id: 1, state: "first", partition: "test"), row(id: 1, state: "second", partition: "test"),
      row(id: 1, state: "parallel", partition: "test", language: "zh-CN")].map.with_index do |item, index|
      EasyAI::Decision::Data::Example.new(item.merge("id" => index.to_s))
    end
    original = EasyAI::Decision::Data::RoutingCorpus.independent_rows(examples)
    reversed = EasyAI::Decision::Data::RoutingCorpus.independent_rows(examples.reverse)

    assert_equal 2, original.size
    assert_equal original.map(&:id).sort, reversed.map(&:id).sort
  end
end
