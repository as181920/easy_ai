require "test_helper"

# Each regression checks one contract across several tensors or translated examples.
# rubocop:disable Minitest/MultipleAssertions

class EvidenceTest < Minitest::Test
  def row(id: "one", index: 1)
    EasyAI::Decision::Data::Example.new({ id: id, group_id: id, language: "en",
      state: "Alice bought a ticket.\nBob did not buy a ticket.", question: "Did Bob buy a ticket?",
      options: [{ id: 0, text: "yes" }, { id: 1, text: "no" }], target: 1, evidence_index: index })
  end

  def configuration(weight = 0.3)
    tiny_config(model: { evidence_head: true }, input: { state_max_tokens: 128 }, training: { evidence_loss_weight: weight, track_coverage: true })
  end

  def test_byte_offsets_match_multilingual_text_and_normal_tokenization
    texts = ["小林。\n小周。", "遅刻しない。\n늦지 않아요.", "لن أتأخر\nJosé arrived."]
    [EasyAI::Tokenizers::ByteBpe.new, EasyAI::Tokenizers::NativeBpe.new.train(texts, vocab_size: 512)].each do |tokenizer|
      texts.each do |text|
        ids, offsets = tokenizer.encode_with_offsets(text)

        assert_equal tokenizer.encode(text), ids
        assert_equal text.bytesize, offsets.last.last
        assert offsets.all? { |start, stop| start >= 0 && stop > start && stop <= text.bytesize }
      end
    end
  end

  def test_gold_annotation_does_not_change_model_inputs_or_outputs
    config, tokenizer = configuration, EasyAI::Tokenizers::ByteBpe.new
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config)
    model = EasyAI::Decision::ChoiceModel.new(config).eval
    annotated = collator.call([row], with_evidence: true)
    changed = collator.call([row(index: 0)], with_evidence: true)
    plain = collator.call([EasyAI::Decision::Data::Example.new(row.to_h.except("evidence_index"))], with_evidence: true)

    assert_equal [1], annotated[:evidence_targets].to_a
    assert_equal [-100], plain[:evidence_targets].to_a
    annotated.except(:evidence_targets).each do |key, value|
      assert_equal value.to_a, changed.fetch(key).to_a
      assert_equal value.to_a, plain.fetch(key).to_a
    end
    Torch.no_grad do
      outputs = model.forward_with_evidence(annotated)

      assert_tensor_close model.call(annotated), outputs.fetch(:logits)
      assert_tensor_close outputs.fetch(:evidence_logits), model.forward_with_evidence(plain).fetch(:evidence_logits)
    end
  end

  def test_evidence_loss_reaches_shared_encoder_and_masks_padded_sentences
    model = EasyAI::Decision::ChoiceModel.new(configuration)
    shorter = EasyAI::Decision::Data::Example.new(row.to_h.merge("state" => "Bob did not buy a ticket.", "evidence_index" => 0))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: model.config)
    batch = collator.call([row, shorter], with_evidence: true)
    outputs = model.forward_with_evidence(batch)
    loss = Torch::NN::Functional.cross_entropy(outputs[:evidence_logits], batch[:evidence_targets])
    loss.backward

    assert_predicate loss.item, :finite?
    assert_equal(-Float::INFINITY, outputs[:evidence_logits].to_a.last.last)
    assert_operator model.encoder.embedding.weight.grad.abs.sum.item, :>, 0
    evidence = model.named_parameters.to_h.select { |name, _tensor| name.start_with?("evidence_head") }

    assert_operator evidence.values.sum { |tensor| tensor.grad.abs.sum.item }, :>, 0
  end

  def test_evidence_supervision_resumes_exactly_and_handles_public_replay
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "rows.jsonl"), [row, row(id: "two", index: 0), example(id: "public")])
      config = configuration.with(model: { dropout: 0.1 })
      factory = ->(path) do
        Torch.manual_seed(83)
        EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new,
          dataset: data, output: path)
      end
      full = factory.call(File.join(dir, "full"))
      full.train(steps: 4)
      partial = factory.call(File.join(dir, "partial"))
      partial.train(steps: 2)
      resumed = EasyAI::Decision::Trainer.resume(partial.last_checkpoint, dataset: data, output: File.join(dir, "resume"), device: "cpu")
      resumed.train(steps: 4)

      full.model.state_dict.each { |name, tensor| assert_tensor_close tensor, resumed.model.state_dict.fetch(name), 1e-7 }
      assert_equal full.state.fetch("coverage"), resumed.state.fetch("coverage")
      assert EasyAI::Decision::Checkpoint.load(resumed.last_checkpoint)[:model].config[:model]["evidence_head"]
      assert File.readlines(File.join(dir, "full/training.jsonl")).map { |line| JSON.parse(line).fetch("train_loss") }.all?(&:finite?)
    end
  end

  def test_generated_labels_and_evidence_reverse_with_facts_and_order
    corpus = EasyAI::Decision::Data::EvidenceCorpus.new
    rows = corpus.examples([0, 1, 0], style: 0)
    keys = { "language" => "zh-CN" }
    original = rows.find { |item| item.slice(*keys.keys) == keys && item.dig("world", "facts") == [true, false] &&
      item.dig("world", "subject") == 1 && item.dig("world", "assertion") && item.dig("world", "order") == 0 && item.dig("world", "distractor") == "none" }

    assert_equal "小林买了票。\n小周没有买票。", original.fetch("state")
    assert_equal "no", original.fetch("target")
    assert_equal 1, original.fetch("evidence_index")
    reversed = rows.find { |item| item.dig("world", "checks", "order") == original.dig("world", "checks", "order") && item != original }

    assert_equal "小周没有买票。\n小林买了票。", reversed.fetch("state")
    assert_equal 0, reversed.fetch("evidence_index")
    assert_equal original.fetch("target"), reversed.fetch("target")
    grouped = rows.group_by { |item| item.dig("world", "checks", "binding") }

    assert grouped.values.all? { |items| items.size == 4 && items.count { |item| item["target"] == "yes" } == 2 }
  end

  def test_generated_splits_isolate_families_and_keep_sanity_inside_training
    Dir.mktmpdir do |dir|
      path = File.join(dir, "data")
      manifest = EasyAI::Decision::Data::EvidenceCorpus.new.write(output: path)
      groups = manifest.fetch("families").transform_values(&:to_set)

      assert groups.values.combination(2).all? { |left, right| (left & right).empty? }
      assert_equal 128, groups.values.sum(&:size)
      train = EasyAI::Decision::Data::Dataset.new(File.join(path, "train.jsonl"))
      sanity = EasyAI::Decision::Data::Dataset.new(File.join(path, "sanity.jsonl"))

      assert_equal 24_576, train.size
      assert_equal 128, sanity.size
      assert_empty sanity.groups - train.groups
      assert_equal %w[en-US zh-CN], sanity.map(&:language).uniq.sort
    end
  end

  def test_invalid_evidence_annotations_and_configuration_fail_early
    assert_raises(ArgumentError) { row(index: -1) }
    assert_raises(ArgumentError) { row(index: 2) }
    assert_raises(ArgumentError) { row(index: "1") }
    assert_raises(ArgumentError) { tiny_config(training: { evidence_loss_weight: 0.1 }) }
    assert_raises(ArgumentError) { configuration.with(model: { encoding_mode: "joint" }) }
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new,
      config: configuration.with(input: { state_max_tokens: 8, truncation: "truncate" }))
    assert_raises(ArgumentError) { collator.call([row], with_evidence: true) }
  end
end
# rubocop:enable Minitest/MultipleAssertions
