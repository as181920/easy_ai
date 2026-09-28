require "test_helper"

class MemoryTest < Minitest::Test
  def test_repeated_choice_validation_releases_temporary_tensors
    assert_validation_releases_tensors(:choice)
  end

  def test_repeated_mlm_validation_releases_temporary_tensors
    assert_validation_releases_tensors(:mlm)
  end

  def test_long_lived_predictor_retains_only_a_bounded_cache
    config = tiny_config(runtime: { cache_entries: 2, candidate_chunk_size: 1 })
    predictor = EasyAI::Decision::Predictor.new(model: EasyAI::Decision::ChoiceModel.new(config),
      tokenizer: EasyAI::Tokenizers::ByteBpe.new, device: "cpu")
    without_automatic_gc do
      4.times { |i| predictor.logits(example(state: "red #{i}")) }
      before = tensor_count
      32.times { |i| predictor.logits(example(state: "blue #{i}")) }

      assert_operator tensor_count, :<=, before + 8
      assert_equal "cpu", predictor.device
    end
  end

  def test_repeated_rotary_relation_evaluation_releases_tensors
    Dir.mktmpdir do |dir|
      rows = EasyAI::Decision::Data::RelationCorpus.new.examples([0, 1, 0], style: 0)
      data = write_dataset(File.join(dir, "relations.jsonl"), rows)
      config = tiny_config(model: { position_encoding: "rotary" }, input: { state_max_tokens: 256, question_option_max_tokens: 128 })
      predictor = EasyAI::Decision::Predictor.new(model: EasyAI::Decision::ChoiceModel.new(config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, device: "cpu")
      evaluator = EasyAI::Decision::RelationEvaluation.new(predictor)
      without_automatic_gc do
        expected = evaluator.evaluate(data, controls: true)
        before = tensor_count
        3.times do
          actual = evaluator.evaluate(data, controls: true)

          assert_equal expected, actual
        end

        assert_operator tensor_count, :<=, before + 8
      end
    end
  end

  private

  def assert_validation_releases_tensors(task)
    Dir.mktmpdir do |dir|
      %w[train validation].each do |split|
        rows = Array.new(split == "train" ? 2 : 16) do |index|
          id = "#{split}-#{index}"
          task == :choice ? example(id: id).to_h : { id: id, group_id: id, text: "red and blue " * (index % 4 + 1) }
        end
        File.write(File.join(dir, "#{split}.jsonl"), rows.map { |row| JSON.generate(row) }.join("\n") + "\n")
      end
      instance = EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(tiny_config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, task: task, output: File.join(dir, "run"),
        dataset: EasyAI::Decision::Data::Dataset.new(File.join(dir, "train.jsonl"), kind: task),
        validation: EasyAI::Decision::Data::Dataset.new(File.join(dir, "validation.jsonl"), kind: task))
      without_automatic_gc do
        first = instance.send(:validation_loss)
        before = tensor_count

        3.times { assert_in_delta first, instance.send(:validation_loss), 1e-8 }

        assert_operator tensor_count, :<=, before + 8
        assert instance.model.parameters.all? { |parameter| parameter.grad.nil? }
      end
    end
  end

  def without_automatic_gc
    GC.start
    previously_disabled = GC.disable
    yield
  ensure
    GC.enable unless previously_disabled
    GC.start
  end

  def tensor_count
    ObjectSpace.each_object(Torch::Tensor).count
  end
end
