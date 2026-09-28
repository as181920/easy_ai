require "test_helper"
require "mocha/minitest"
require "faraday"

class SemanticDataTest < Minitest::Test
  def test_shared_material_and_official_dev_cannot_leak_into_training
    state = "shared material " * 4
    rows = [row(origin: "a", state: state), row(origin: "b", state: state, partition: "dev"), row(origin: "b", state: "other")]
    result = EasyAI::Decision::Data::SemanticCorpus.new(rows).split_rows

    assert_equal 1, result.size
    assert_equal "test", result.first.last
    assert_equal "dev", result.first.first[:partition]
  end

  def test_short_generic_answers_do_not_join_unrelated_questions
    rows = [row(origin: "a", state: "yes", question: "Ready?"), row(origin: "b", state: "yes", question: "Done?")]
    result = EasyAI::Decision::Data::SemanticCorpus.new(rows).split_rows

    assert_equal 2, result.map { |_item, group, _split| group }.uniq.size
  end

  def test_preparation_writes_supervision_and_corpus_without_cross_split_groups
    Dir.mktmpdir do |dir|
      rows = Array.new(100) { |i| row(origin: "origin-#{i}", state: "statement #{i}", question: "Question #{i}", partition: i < 10 ? "dev" : "train") }
      config = tiny_config(model: { vocab_size: 300 })
      output = File.join(dir, "data")
      result = EasyAI::Decision::Data::SemanticCorpus.new(rows).write(output: output, config: config,
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, limit: 3)
      datasets = %w[train validation calibration test].map { |split| EasyAI::Decision::Data::Dataset.new(File.join(output, "#{split}.jsonl")) }
      EasyAI::Decision::Data::Dataset.assert_disjoint!(*datasets)
      corpus = EasyAI::Decision::Data::Dataset.new(File.join(output, "corpus.jsonl"), kind: :mlm)

      assert_equal datasets.first.groups, corpus.groups
      assert_equal "yes", datasets.first[0].target
      assert_equal 3, datasets.last.groups.size
      assert_operator result["rows"]["train/fixture"], :>, 0
    end
  end

  def test_faraday_download_streams_and_checks_the_digest
    Dir.mktmpdir do |dir|
      connection = Object.new
      request = Struct.new(:options).new(Struct.new(:on_data).new)
      connection.define_singleton_method(:get) do |_url, &block|
        block.call(request)
        request.options.on_data.call("hello", 5, nil)
        Struct.new(:status, :headers) { def success? = true }.new(200, {})
      end
      Faraday.stubs(:new).returns(connection)
      path = File.join(dir, "source.jsonl")
      result = EasyAI::Decision::Data::Download.fetch("https://example.test/data", path, expected_sha256: Digest::SHA256.hexdigest("hello"))

      assert_equal "hello", File.read(path)
      assert_equal 5, result["bytes"]
      refute_path_exists "#{path}.part"
    end
  end

  def test_failed_download_does_not_leave_a_completed_file
    Dir.mktmpdir do |dir|
      connection = Object.new
      connection.define_singleton_method(:get) { |_url, &_block| Struct.new(:status, :headers) { def success? = false }.new(503, {}) }
      Faraday.stubs(:new).returns(connection)
      path = File.join(dir, "source.jsonl")

      assert_raises(RuntimeError) { EasyAI::Decision::Data::Download.fetch("https://example.test/data", path) }
      refute_path_exists path
      refute_path_exists "#{path}.part"
    end
  end

  def test_checksum_failure_and_size_limit_remove_partial_files
    Dir.mktmpdir do |dir|
      stream_response("hello")
      path = File.join(dir, "source.jsonl")

      assert_raises(RuntimeError) { EasyAI::Decision::Data::Download.fetch("https://example.test/data", path, expected_sha256: "wrong") }
      refute_path_exists path
      assert_raises(RuntimeError) { EasyAI::Decision::Data::Download.fetch("https://example.test/data", path, max_mib: 0) }
      refute_path_exists "#{path}.part"
    end
  end

  def test_https_redirect_cannot_downgrade_to_plain_http
    Dir.mktmpdir do |dir|
      stream_response("redirect", status: 302, headers: { "location" => "http://example.test/insecure" })
      path = File.join(dir, "source.jsonl")

      assert_raises(ArgumentError) { EasyAI::Decision::Data::Download.fetch("https://example.test/data", path) }
      refute_path_exists path
      refute_path_exists "#{path}.part"
    end
  end

  private

  def stream_response(body, status: 200, headers: {})
    connection = Object.new
    connection.define_singleton_method(:get) do |_url, &block|
      request = Struct.new(:options).new(Struct.new(:on_data).new)
      block.call(request)
      request.options.on_data.call(body, body.bytesize, nil)
      Struct.new(:status, :headers) { def success? = status == 200 }.new(status, headers)
    end
    Faraday.stubs(:new).returns(connection)
  end

  def row(origin:, state:, question: "True?", partition: "train")
    { source: "fixture", language: "en", origin: origin, partition: partition, state: state, question: question,
      target: "yes", options: { "yes" => "yes", "no" => "no" } }
  end
end
