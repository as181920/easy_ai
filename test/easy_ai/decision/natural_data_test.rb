require "test_helper"
require "csv"

class NaturalDataTest < Minitest::Test
  def row(id, state, language: "en-US", partition: "train", target: "0")
    EasyAI::Decision::Data::NaturalAdapter.build("test", language, id, state, "Which?", { "0" => "First", "1" => "Second" }, target, partition)
  end

  def splits(rows, blocked: Set.new)
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: tiny_config(input: { state_max_tokens: 128 }))
    EasyAI::Decision::Data::NaturalCorpus.new(rows, blocked: blocked, collator: collator).splits
  end

  def test_parallel_ids_and_duplicate_material_never_cross_splits
    rows = [row("a", "English sentence"), row("a", "中文句子", language: "zh-CN"), row("b", "English sentence")]
    result = splits(rows)
    nonempty = result.reject { |_split, items| items.empty? }

    assert_equal 1, nonempty.size
    assert_equal 2, nonempty.values.first.size
    assert_equal 1, nonempty.values.first.map { |item| item.fetch("group_id") }.uniq.size
  end

  def test_blocking_one_translation_excludes_whole_component
    rows = [row("a", "English sentence"), row("a", "中文句子", language: "zh-CN")]
    blocked = [EasyAI::Decision::Data::NaturalCorpus.material("English sentence")].to_set

    assert splits(rows, blocked: blocked).values.all?(&:empty?)
  end

  def test_official_dev_is_never_training_and_overlapping_train_is_removed
    rows = [row("a", "Shared", partition: "train"), row("b", "Shared", partition: "dev")]
    result = splits(rows)

    assert_empty result.fetch("train")
    assert_equal 1, result.values.flatten.size
    assert_equal "dev", result.values.flatten.first.fetch("partition")
  end

  def test_candidates_and_split_assignment_do_not_depend_on_target
    original = splits([row("a", "A state", target: "0")])
    changed = splits([row("a", "A state", target: "1")])

    assert_equal original.transform_values { |items| items.map { |item| item.except("target") } }, changed.transform_values { |items| items.map { |item| item.except("target") } }
    assert_equal 2, original.values.flatten.first.fetch("options").size
  end

  def test_conflicting_duplicate_labels_exclude_connected_translations
    rows = [row("a", "Same", target: "0"), row("b", "Same", target: "1"), row("a", "中文", language: "zh-CN")]

    assert splits(rows).values.all?(&:empty?)
  end

  def test_test_and_heldout_partitions_cannot_enter_training_corpus
    assert_raises(ArgumentError) { splits([row("a", "State", partition: "test")]) }
    assert_raises(ArgumentError) { splits([row("a", "State", partition: "heldout")]) }
  end

  def test_news_csv_handles_quotes_and_retains_all_four_candidates
    Dir.mktmpdir do |directory|
      path = File.join(directory, "train.csv")
      File.write(path, CSV.generate_line(["2", 'A "quoted", title', "Details"]))
      item = EasyAI::Decision::Data::NaturalAdapter.news(path).first

      assert_equal 'A "quoted", title Details', item.fetch("state")
      assert_equal "1", item.fetch("target")
      assert_equal 4, item.fetch("options").size
      assert_equal "train", item.fetch("partition")
    end
  end
end
