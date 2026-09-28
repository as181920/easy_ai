require "test_helper"
require "zlib"
require "rubygems/package"

class DataTest < Minitest::Test
  def test_parallel_translations_cannot_cross_splits
    Dir.mktmpdir do |dir|
      first = write_dataset(File.join(dir, "train.jsonl"), [example(id: "same", language: "zh")])
      second = write_dataset(File.join(dir, "test.jsonl"), [example(id: "same", language: "en")])
      assert_raises(ArgumentError) { EasyAI::Decision::Data::Dataset.assert_disjoint!(first, second) }
    end
  end

  def test_masking_is_repeatable_and_excludes_special_tokens
    masker = EasyAI::Decision::Data::Masking.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: tiny_config)
    rows = [{ "text" => "中英文 hello" }]
    first = masker.call(rows, seed: 15)
    second = masker.call(rows, seed: 15)

    assert_equal first[:ids].to_a, second[:ids].to_a
    assert_equal first[:positions].to_a, second[:positions].to_a
    assert first[:targets].to_a.all? { |i| i >= 6 }
  end

  def test_massive_keeps_original_groups_across_languages
    Dir.mktmpdir do |dir|
      archive = File.join(dir, "massive.tar.gz")
      make_archive(archive)
      out = File.join(dir, "prepared")
      result = EasyAI::Decision::Data::Adapters::Massive.prepare(archive: archive, output: out, locales: %w[zh-CN en-US], candidates: 2)
      splits = %w[train validation calibration test].map { |name| EasyAI::Decision::Data::Dataset.new(File.join(out, "#{name}.jsonl")) }
      EasyAI::Decision::Data::Dataset.assert_disjoint!(*splits)

      assert_equal 2, result["counts"]["train"]["zh-CN"]
      assert_equal 2, splits.first.groups.size
      assert File.file?(File.join(out, "corpus-validation.jsonl"))
      assert_includes File.read(File.join(out, "LICENSE")), "CC-BY"
    end
  end

  def test_only_capacity_failures_are_recoverable
    policy = EasyAI::Runtime::DevicePolicy.new(requested: "cpu")

    assert policy.recoverable?(Torch::Error.new("CUDA out of memory"))
    refute policy.recoverable?(Torch::Error.new("shape mismatch"))
    refute policy.recoverable?(ArgumentError.new("CUDA out of memory"))
  end

  def test_negative_resampling_keeps_target_and_is_reproducible
    rows = [example, example(id: "green", target: "g", options: [{ id: "g", text: "green" }, { id: "r", text: "red" }])]
    sampler = EasyAI::Decision::Data::CandidateSampler.new(rows)
    left = sampler.call(rows.first, rng: Random.new(42))
    right = sampler.call(rows.first, rng: Random.new(42))

    assert_equal left.to_h, right.to_h
    assert_equal "r", left.target
    assert_equal 2, left.options.map { |option| option["id"] }.uniq.size
    assert_includes left.options.map { |option| option["id"] }, left.target
  end

  def test_resampling_rejects_candidate_ids_with_changing_meanings
    rows = [example, example(options: [{ id: "r", text: "wrong meaning" }, { id: "b", text: "blue" }])]

    assert_raises(ArgumentError) { EasyAI::Decision::Data::CandidateSampler.new(rows) }
  end

  def test_training_limit_can_grow_without_changing_held_out_rows
    Dir.mktmpdir do |dir|
      archive = File.join(dir, "massive.tar.gz")
      make_archive(archive)
      paths = [File.join(dir, "small"), File.join(dir, "expanded")]
      paths.zip([1, 2]).each do |path, train_limit|
        EasyAI::Decision::Data::Adapters::Massive.prepare(archive: archive, output: path,
          locales: ["en-US"], candidates: 2, limit: 1, train_limit: train_limit)
      end

      assert_equal [1, 2], paths.map { |path| EasyAI::Decision::Data::Dataset.new(File.join(path, "train.jsonl")).size }
      %w[validation calibration test].each do |split|
        assert_equal File.read(File.join(paths[0], "#{split}.jsonl")), File.read(File.join(paths[1], "#{split}.jsonl"))
      end
    end
  end

  private

  def make_archive(path)
    buffer = StringIO.new("".b)
    Gem::Package::TarWriter.new(buffer) do |tar|
      %w[zh-CN en-US].each do |locale|
        rows = [0, 1].map { |i| { id: i.to_s, locale: locale, partition: "train", intent: "label_#{i}", utt: "sample #{i}" } }
        rows += (2..15).map { |i| { id: i.to_s, locale: locale, partition: "dev", intent: "label_0", utt: "dev #{i}" } }
        rows << { id: "16", locale: locale, partition: "test", intent: "label_0", utt: "test" }
        text = rows.map { |row| JSON.generate(row) }.join("\n")
        tar.add_file_simple("1.1/data/#{locale}.jsonl", 0o644, text.bytesize) { |file| file.write(text) }
      end
    end
    Zlib::GzipWriter.open(path) { |gzip| gzip.write(buffer.string) }
  end
end
