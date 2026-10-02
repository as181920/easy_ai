require "test_helper"

class ReleaseTest < Minitest::Test
  class FakePredictor
    def probabilities(state:, question:, options:)
      { "probabilities" => options.each_with_index.to_h { |option, index| [(option[:id] || option["id"]).to_s, index.zero? ? 0.95 : 0.05 / (options.size - 1)] },
        "calibrated" => true, "temperature" => 1.0, "device" => "cpu", "truncated_segments" => 0 }
    end
  end

  def release
    EasyAI::Decision::Release.new(predictor: FakePredictor.new,
      metadata: { "version" => "0.1", "languages" => %w[en-US zh-CN].to_h { |language| [language, { "available" => true, "threshold" => 0.9 }] } })
  end

  def test_evaluated_profile_can_suggest_an_option_with_numeric_ids
    model = release
    options = model.options("zh-CN").each_with_index.map { |option, index| option.merge(id: index) }
    result = model.probabilities(state: "明天七点叫我起床", question: "这项请求属于哪个领域？", options: options, language: "zh-CN")

    assert result.fetch("supported_profile")
    refute result.fetch("review_required")
    assert_equal "0", result.fetch("suggested_option")
  end

  def test_arbitrary_candidates_still_return_probabilities_but_do_not_claim_support
    result = release.probabilities(state: "我要迟到了", question: "会迟到么？", options: [{ id: 0, text: "会" }, { id: 1, text: "不会" }], language: "zh-CN")

    refute result.fetch("supported_profile")
    assert result.fetch("review_required")
    assert_equal %w[0 1], result.fetch("probabilities").keys
  end

  def test_route_uses_bilingual_question_and_all_domains
    result = release.route(state: "Wake me at seven", language: "en-US")

    assert result.fetch("supported_profile")
    assert_equal 18, result.fetch("probabilities").size
    assert_raises(ArgumentError) { release.route(state: "hello", language: "ja-JP") }
  end

  def test_failed_acceptance_cannot_be_published
    Dir.mktmpdir do |root|
      File.write(File.join(root, "acceptance.json"), JSON.generate("passed" => false))
      File.write(File.join(root, "policy.json"), "{}")
      File.write(File.join(root, "protocol.json"), "{}")

      assert_raises(ArgumentError) { EasyAI::Decision::Release.publish(run: root, output: File.join(root, "release")) }
      refute_path_exists File.join(root, "release")
    end
  end

  def test_published_bundle_is_portable_and_checksums_are_enforced
    check_bundle(preview: false)
  end

  def test_preview_preserves_failed_quality_result_and_is_portable
    check_bundle(preview: true)
  end

  def check_bundle(preview:)
    Dir.mktmpdir do |root|
      source = File.join(root, "source")
      FileUtils.mkdir_p(source)
      model = EasyAI::Decision::ChoiceModel.new(tiny_config)
      checkpoint = EasyAI::Decision::Checkpoint.save(File.join(source, "calibrated"), model: model,
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, calibration: { "temperature" => 1.0 })
      sha = Digest::SHA256.file(File.join(checkpoint, "weights.pt")).hexdigest
      languages = %w[en-US zh-CN]
      policy = { "checkpoint" => checkpoint, "weights_sha256" => sha, "temperature" => 1.0,
        "languages" => languages.to_h { |language| [language, { "available" => true, "threshold" => 0.9 }] } }
      protocol = { "requirements" => EasyAI::Decision::ReleasePolicy::REQUIREMENTS, "languages" => languages,
        "profile" => "bilingual-request-domains", "scope" => "Packaging fixture, no semantic quality claim",
        "source" => {}, "parent_sha256" => "fixture" }
      File.write(File.join(source, "policy.json"), JSON.generate(policy))
      File.write(File.join(source, "protocol.json"), JSON.generate(protocol))
      metrics = { "count" => 600, "accuracy" => 1.0, "balanced_accuracy" => 1.0, "coverage" => 1.0,
        "accepted_count" => 600, "accepted_accuracy" => 1.0, "accepted_accuracy_lower_95" => 0.99,
        "recall" => (0...18).to_h { |index| [index.to_s, 1.0] } }
      metrics["accuracy"] = 0.78 if preview
      acceptance = { "passed" => !preview, "by_language" => languages.to_h { |language| [language, metrics] },
        "policy_sha256" => Digest::SHA256.file(File.join(source, "policy.json")).hexdigest,
        "protocol_sha256" => Digest::SHA256.file(File.join(source, "protocol.json")).hexdigest }
      File.write(File.join(source, "acceptance.json"), JSON.generate(acceptance))
      File.write(File.join(source, "runtime.json"), JSON.generate("passed" => true, "weights_sha256" => sha))
      destination = File.join(root, "bundle")
      if preview
        assert_raises(ArgumentError) { EasyAI::Decision::Release.publish(run: source, output: destination) }
      end
      EasyAI::Decision::Release.publish(run: source, output: destination, preview: preview)
      FileUtils.rm_rf(source)
      loaded = EasyAI::Decision::Release.load(destination, device: "cpu")
      result = loaded.probabilities(state: "test", question: "q", options: [{ id: 0, text: "yes" }, { id: 1, text: "no" }], language: "en-US")

      assert_equal preview ? "preview" : "stable", result.fetch("release_status")
      assert_equal !preview, loaded.metadata.fetch("acceptance_passed")
      assert_equal !preview, JSON.parse(File.read(File.join(destination, "acceptance.json"))).fetch("passed")
      assert_equal %w[0 1], result.fetch("probabilities").keys
      assert result.fetch("review_required")
      File.open(File.join(destination, "release.json"), "a") { |file| file.write(" ") }

      assert_raises(ArgumentError) { EasyAI::Decision::Release.load(destination, device: "cpu") }
    end
  end
end
