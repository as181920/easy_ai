require "test_helper"

class CliTest < Minitest::Test
  def test_complete_local_training_calibration_prediction_flow
    Dir.mktmpdir do |dir|
      paths = %w[train calibration test].to_h do |split|
        path = File.join(dir, "#{split}.jsonl")
        write_dataset(path, [example(id: "#{split}-red"), example(id: "#{split}-blue", state: "blue", target: "b")])
        [split, path]
      end
      config = File.join(dir, "config.yml")
      File.write(config, YAML.dump(tiny_config.to_h))
      tokenizer_path = File.join(dir, "tokenizer.json")
      EasyAI::Tokenizers::ByteBpe.new.save(tokenizer_path)
      run = File.join(dir, "run")
      trained = cli("train", "--config", config, "--tokenizer", tokenizer_path, "--data", paths["train"], "--output", run, "--steps", "2")

      assert_equal 2, trained["step"]
      calibrated = File.join(dir, "calibrated")
      cli("calibrate", "--checkpoint", run, "--data", paths["calibration"], "--output", calibrated, "--device", "cpu")
      evaluated = cli("evaluate", "--checkpoint", calibrated, "--data", paths["test"], "--device", "cpu")

      assert_equal 2, evaluated["count"]
      assert evaluated["calibrated"]
      input = File.join(dir, "request.json")
      File.write(input, JSON.generate(example.to_h))
      predicted = cli("predict", "--checkpoint", calibrated, "--input", input, "--device", "cpu")

      assert_in_delta 1.0, predicted["probabilities"].values.sum, 1e-12
    end
  end

  def test_calibration_rejects_training_data
    Dir.mktmpdir do |dir|
      data = File.join(dir, "train.jsonl")
      write_dataset(data, [example(id: "seen")])
      EasyAI::Decision::Checkpoint.save(dir, model: EasyAI::Decision::ChoiceModel.new(tiny_config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, training_state: { "groups" => { "train" => ["seen"] } })
      out, err = StringIO.new, StringIO.new
      code = EasyAI::Decision::Cli.run(["calibrate", "--checkpoint", dir, "--data", data, "--output", File.join(dir, "cal")], out: out, err: err)

      assert_equal 1, code
      assert_includes err.string, "overlaps"
    end
  end

  def test_mlm_validation_and_transfer_to_choices_preserve_split_history
    Dir.mktmpdir do |dir|
      paths = %w[train validation].to_h do |split|
        path = File.join(dir, "#{split}.jsonl")
        row = { id: split, group_id: split, language: "zh-CN", text: "你好，学习神经网络。" }
        File.write(path, JSON.generate(row) + "\n")
        [split, path]
      end
      config = File.join(dir, "config.yml")
      File.write(config, YAML.dump(tiny_config.to_h))
      tokenizer = File.join(dir, "tokenizer.json")
      EasyAI::Tokenizers::ByteBpe.new.save(tokenizer)
      mlm = File.join(dir, "mlm")
      cli("pretrain", "--config", config, "--tokenizer", tokenizer, "--data", paths["train"],
        "--validation", paths["validation"], "--output", mlm, "--steps", "2")
      checkpoint = EasyAI::Decision::Checkpoint.load(mlm)

      assert_equal "mlm", checkpoint[:metadata]["training"]["task"]
      assert_predicate checkpoint[:metadata]["training"]["history"].last["validation_loss"], :finite?
      paths.each do |split, path|
        write_dataset(path, [example(id: split)])
      end
      choice = File.join(dir, "choice")
      result = cli("train", "--init", mlm, "--data", paths["train"], "--validation", paths["validation"],
        "--output", choice, "--steps", "2")
      trained = EasyAI::Decision::Checkpoint.load(choice)

      assert_equal 2, result["step"]
      assert_equal "choice", trained[:metadata]["training"]["task"]
      assert_equal({ "train" => ["train"], "validation" => ["validation"] }, trained[:metadata]["training"]["groups"])
    end
  end

  def test_unknown_command_has_nonzero_exit
    assert_equal 1, EasyAI::Decision::Cli.run(["missing"], out: StringIO.new, err: StringIO.new)
  end

  private

  def cli(*args)
    out, err = StringIO.new, StringIO.new
    code = EasyAI::Decision::Cli.run(args, out: out, err: err)
    raise "CLI failed: #{err.string}" unless code.zero?
    JSON.parse(out.string)
  end
end
