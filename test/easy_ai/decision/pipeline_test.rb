require "test_helper"
require "zlib"
require "rubygems/package"

class PipelineTest < Minitest::Test
  def test_one_command_trains_from_archive_and_creates_reports
    Dir.mktmpdir do |dir|
      archive = File.join(dir, "massive.tar.gz")
      make_archive(archive)
      config = File.join(dir, "tiny.yml")
      File.write(config, YAML.dump(tiny_config.to_h))
      output = File.join(dir, "run with spaces")
      out, progress = StringIO.new, StringIO.new
      code = EasyAI::Decision::Cli.run(["pipeline", "--config", config, "--archive", archive, "--output", output,
        "--locales", "en-US", "--candidates", "2", "--mlm-steps", "2", "--choice-steps", "3", "--device", "cpu"], out: out, err: progress)
      result = JSON.parse(out.string)

      assert_equal [0, "complete"], [code, result["status"]]
      assert_equal %w[prepare tokenizer mlm choice diagnostic calibration test], result["stages"]
      assert_equal [2, 3], %w[mlm choice].map { |stage| File.foreach(File.join(output, stage, "training.jsonl")).count }
      assert_equal "\x89PNG\r\n\x1a\n".b, File.binread(result["report"]["loss_png"], 8)
      assert_match(/\[choice\].*3\/3.*device=cpu/, progress.string)
    end
  end

  def test_preflight_rejects_mlm_leakage_before_starting_training
    Dir.mktmpdir do |dir|
      %w[train validation calibration test].each do |split|
        write_dataset(File.join(dir, "#{split}.jsonl"), [example(id: split)])
      end
      %w[corpus corpus-validation].zip(%w[test validation]).each do |name, group|
        File.write(File.join(dir, "#{name}.jsonl"), JSON.generate(id: group, group_id: group, text: "red") + "\n")
      end
      tokenizer = File.join(dir, "tokenizer.json")
      EasyAI::Tokenizers::ByteBpe.new.save(tokenizer)
      output = File.join(dir, "run")
      pipeline = EasyAI::Decision::Pipeline.new({ data: dir, tokenizer: tokenizer, output: output, device: "cpu" }, progress: StringIO.new)

      error = assert_raises(ArgumentError) { pipeline.run }

      assert_includes error.message, "MLM corpus groups overlap"
      assert_equal "failed", JSON.parse(File.read(File.join(output, "summary.json")))["status"]
      refute_path_exists File.join(output, "mlm")
    end
  end

  def test_pipeline_can_train_supervision_from_random_weights_without_mlm
    Dir.mktmpdir do |dir|
      archive = File.join(dir, "massive.tar.gz")
      make_archive(archive)
      config = File.join(dir, "tiny.yml")
      File.write(config, YAML.dump(tiny_config.to_h))
      result = EasyAI::Decision::Pipeline.new({ config: config, archive: archive, output: File.join(dir, "run"),
        locales: ["en-US"], candidates: 2, mlm_steps: 0, choice_steps: 2, device: "cpu" }, progress: StringIO.new).run

      refute_includes result["stages"], "mlm"
      assert_nil result["selection"]["mlm"]
      assert_path_exists File.join(result["output"], "stage-results/diagnostic.json")
      assert_includes File.read(result["report"]["html"]), "状态依赖诊断"
    end
  end

  def test_report_can_plot_one_step_and_uses_latest_retried_observation
    Dir.mktmpdir do |dir|
      FileUtils.mkdir_p(File.join(dir, "mlm"))
      rows = [{ step: 1, train_loss: 2.0 }, { step: 1, train_loss: 1.5 }]
      File.write(File.join(dir, "mlm/training.jsonl"), rows.map { |row| JSON.generate(row) }.join("\n"))
      File.write(File.join(dir, "mlm/metrics.jsonl"), JSON.generate(step: 1, validation_loss: 1.7))
      report = EasyAI::Decision::TrainingReport.new(dir, out: StringIO.new).write

      assert_equal "1\t1.5\n", File.read(File.join(dir, "report/mlm-train.dat"))
      assert_includes File.read(report["console"]), "MLM loss"
      assert_includes File.read(report["html"]), "Test evaluation has not completed"
    end
  end

  private

  def make_archive(path)
    buffer = StringIO.new("".b)
    Gem::Package::TarWriter.new(buffer) do |tar|
      rows = [0, 1].map { |i| { id: i.to_s, partition: "train", intent: "label_#{i}", utt: "sample #{i}" } }
      rows += (2..15).map { |i| { id: i.to_s, partition: "dev", intent: "label_0", utt: "dev #{i}" } }
      rows << { id: "16", partition: "test", intent: "label_0", utt: "test" }
      text = rows.map { |row| JSON.generate(row) }.join("\n")
      tar.add_file_simple("1.1/data/en-US.jsonl", 0o644, text.bytesize) { |file| file.write(text) }
    end
    Zlib::GzipWriter.open(path) { |gzip| gzip.write(buffer.string) }
  end
end
