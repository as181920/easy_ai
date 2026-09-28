require "test_helper"
require "faraday"

class DistillationTest < Minitest::Test
  class Teacher
    attr_accessor :fail_after
    attr_reader :calls

    def initialize(kind: "text")
      @kind, @calls = kind, 0
    end

    def signature
      { "backend" => "test_fixture", "revision" => "1", "capabilities" => [@kind] }
    end

    def call(_request)
      raise "injected interruption" if fail_after && calls >= fail_after
      @calls += 1
      @kind == "text" ? { "kind" => "text", "text" => '{"answer":0}' } :
        { "kind" => @kind, "temperature" => 1, "scoring_protocol" => "test_fixture",
          "probabilities" => { "r" => 0.8, "b" => 0.2 } }
    end
  end

  def test_content_is_a_label_not_generated_confidence_and_gold_is_not_sent
    adapter = EasyAI::Decision::DistillationAdapter.new
    row = example(target: "b")
    request = adapter.request(row)
    sent = JSON.parse(request.fetch("messages").last.fetch("content"))

    refute sent.key?("target")
    assert_equal({ "kind" => "label", "target" => "r" }, adapter.parse(row, { "kind" => "text", "text" => '{"answer":0}' }))
    ['{"answer":0,"confidence":0.9}', '{"answer":"0"}', '{"answer":9}', "null"].each do |text|
      assert_raises(ArgumentError) { adapter.parse(row, { "kind" => "text", "text" => text }) }
    end
  end

  def test_task_prompt_uses_source_semantics_without_gold_or_label_position
    adapter = EasyAI::Decision::DistillationAdapter.new(profile: "task_specific")
    EasyAI::Decision::DistillationAdapter::TASK_INSTRUCTIONS.each do |source, instruction|
      row = EasyAI::Decision::Data::Example.new(example.to_h.merge("source" => source))
      changed_gold = EasyAI::Decision::Data::Example.new(row.to_h.merge("target" => "b"))
      reversed = EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse))

      assert_includes adapter.request(row).fetch("messages").first.fetch("content"), instruction
      assert_equal adapter.request(row), adapter.request(changed_gold)
      assert_equal adapter.identity(row), adapter.identity(reversed)
      assert_equal({ "kind" => "label", "target" => "b" }, adapter.parse(reversed, { "kind" => "text", "text" => '{"answer":0}' }))
    end
  end

  def test_task_profile_rejects_unknown_tasks_and_changed_definitions
    adapter = EasyAI::Decision::DistillationAdapter.new(profile: "task_specific")

    assert_raises(ArgumentError) { adapter.request(example) }
    assert_raises(ArgumentError) { EasyAI::Decision::DistillationAdapter.new(profile: "typo") }
    assert_raises(ArgumentError) { EasyAI::Decision::DistillationAdapter.from_signature(adapter.signature.merge("prompt" => "changed")) }
    %w[generic task_specific].each do |profile|
      signature = EasyAI::Decision::DistillationAdapter.new(profile: profile).signature

      assert_equal signature, EasyAI::Decision::DistillationAdapter.from_signature(signature).signature
    end
  end

  def test_task_profile_survives_artifacts_and_rejects_changed_source_or_profile
    Dir.mktmpdir do |dir|
      row = EasyAI::Decision::Data::Example.new(example.to_h.merge("source" => "BoolQ"))
      data = write_dataset(File.join(dir, "data.jsonl"), [row])
      adapter = EasyAI::Decision::DistillationAdapter.new(profile: "task_specific")
      teacher = Teacher.new
      output = File.join(dir, "teacher")
      artifact = EasyAI::Distillation::Collector.new(teacher: teacher, adapter: adapter,
        output: output, purpose: "train", progress: nil).run(data)
      supervision = EasyAI::Decision::DistillationSupervision.new(artifact: artifact, dataset: data)
      changed = EasyAI::Decision::Data::Example.new(row.to_h.merge("source" => "OCNLI"))

      assert_in_delta Math.log(2), supervision.loss(Torch.zeros([1, 2]), [row]).item, 1e-6
      assert_raises(ArgumentError) { supervision.loss(Torch.zeros([1, 2]), [changed]) }
      assert_raises(ArgumentError) do
        EasyAI::Distillation::Collector.new(teacher: teacher, adapter: EasyAI::Decision::DistillationAdapter.new,
          output: output, purpose: "train", progress: nil).run(data)
      end
      assert_equal 1, teacher.calls
    end
  end

  def test_collector_resumes_without_requerying_and_checks_configuration_and_records
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "data.jsonl"), [example(id: "a"), example(id: "b")])
      teacher = Teacher.new
      teacher.fail_after = 1
      collector = EasyAI::Distillation::Collector.new(teacher: teacher, adapter: EasyAI::Decision::DistillationAdapter.new,
        output: File.join(dir, "artifact"), purpose: "train", progress: nil)

      assert_raises(RuntimeError) { collector.run(data) }
      refute_path_exists File.join(dir, "artifact/manifest.json")
      teacher.fail_after = nil
      artifact = collector.run(data)

      assert_equal artifact.fingerprint, collector.run(data).fingerprint
      assert_equal 2, teacher.calls
      changed = write_dataset(File.join(dir, "changed.jsonl"), [example(id: "a", state: "different")])
      assert_raises(ArgumentError) { collector.run(changed) }
    end
  end

  def test_completed_artifact_rejects_corrupt_records
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "data.jsonl"), [example])
      artifact = collect(dir, data, Teacher.new)
      path = artifact.path
      File.open(File.join(path, "records.jsonl"), "a") { |file| file.puts("{}") }

      assert_raises(ArgumentError) { EasyAI::Distillation::Artifact.new(path) }
    end
  end

  def test_kl_matches_scalar_reference_and_padding_has_no_gradient
    logits = Torch.tensor([[1.0, -1.0, -Float::INFINITY], [0.0, 1.0, 2.0]]).detach.requires_grad!(true)
    distributions = [[0.0, 1.0], [0.2, 0.3, 0.5]]
    temperature = 2.0
    expected = distributions.each_with_index.map do |values, i|
      target = values.map { |p| p**(1.0 / temperature) }
      target = target.map { |p| p / target.sum }
      scores = [[1.0, -1.0], [0.0, 1.0, 2.0]][i].map { |s| Math.exp(s / temperature) }
      total = scores.sum
      target.each_with_index.sum { |p, j| p.zero? ? 0 : p * Math.log(p / (scores[j] / total)) } * temperature**2
    end.sum / 2
    loss = EasyAI::Distillation::Losses::SoftTargets.call(logits, distributions, temperature: temperature)
    loss.backward

    assert_in_delta expected, loss.item, 1e-6
    assert_in_delta 0, logits.grad[0][2].item
    assert logits.grad.to_a.flatten.all?(&:finite?)
    reversed = Torch.tensor([[-1.0, 1.0], [2.0, 1.0]])
    forward = Torch.tensor([[1.0, -1.0], [1.0, 2.0]])
    targets = [[0.8, 0.2], [0.4, 0.6]]

    assert_in_delta EasyAI::Distillation::Losses::SoftTargets.call(forward, targets).item,
      EasyAI::Distillation::Losses::SoftTargets.call(reversed, targets.map(&:reverse)).item, 1e-6
  end

  def test_supervision_aligns_by_candidate_identity_and_rejects_development_artifacts
    Dir.mktmpdir do |dir|
      row = example
      data = write_dataset(File.join(dir, "data.jsonl"), [row])
      artifact = collect(dir, data, Teacher.new(kind: "candidate_probabilities"))
      supervision = EasyAI::Decision::DistillationSupervision.new(artifact: artifact, dataset: data)
      reordered = EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse))

      assert_in_delta supervision.loss(Torch.tensor([[1.0, -1.0]]), [row]).item,
        supervision.loss(Torch.tensor([[-1.0, 1.0]]), [reordered]).item, 1e-6
      assert_raises(ArgumentError) { supervision.loss(Torch.tensor([[1.0, -1.0]]), [example(state: "other")]) }
      development = collect(dir, data, Teacher.new, purpose: "development")
      assert_raises(ArgumentError) { EasyAI::Decision::DistillationSupervision.new(artifact: development, dataset: data) }
    end
  end

  def test_hard_teacher_loss_and_validation_are_separate_and_resume_is_exact
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "data.jsonl"), [example(id: "a"), example(id: "b", target: "b")])
      artifact = collect(dir, data, Teacher.new)
      supervision = EasyAI::Decision::DistillationSupervision.new(artifact: artifact, dataset: data, weight: 0.3)
      Torch.manual_seed(42)
      model = EasyAI::Decision::ChoiceModel.new(tiny_config(model: { dropout: 0.1 }))
      trainer = EasyAI::Decision::Trainer.new(model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new,
        dataset: data, output: File.join(dir, "train"), distillation: supervision)
      trainer.train(steps: 2)
      path = trainer.last_checkpoint
      trainer.train(steps: 4)
      resumed = EasyAI::Decision::Trainer.resume(path, dataset: data, output: File.join(dir, "resume"), distillation: supervision)
      resumed.train(steps: 4)

      trainer.model.state_dict.each { |name, tensor| assert_tensor_close tensor, resumed.model.state_dict.fetch(name), 1e-7 }
      assert_raises(ArgumentError) { EasyAI::Decision::Trainer.resume(path, dataset: data, output: File.join(dir, "missing")) }
      changed = EasyAI::Decision::DistillationSupervision.new(artifact: artifact, dataset: data, weight: 0.4)
      assert_raises(ArgumentError) { EasyAI::Decision::Trainer.resume(path, dataset: data, output: File.join(dir, "changed"), distillation: changed) }
      model = trainer.model
      model.eval
      batch = EasyAI::Decision::Data::Collator.new(tokenizer: trainer.tokenizer, config: model.config).call(data.to_a)
      gold = Torch::NN::Functional.cross_entropy(model.call(batch), batch[:targets]).item

      assert_in_delta gold, trainer.send(:loss_for, data.to_a, seed: 0).item, 1e-6
      assert_operator trainer.send(:loss_for, data.to_a, seed: 0, teacher: true).item, :>, gold
    end
  end

  def test_http_teacher_has_bounded_retry_and_rejects_truncated_answers
    attempts = 0
    stubs = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/chat/completions") do |env|
        attempts += 1
        body = JSON.parse(env.body)

        assert_equal "fixture", body.fetch("model")
        if attempts == 1
          [503, {}, "unavailable"]
        else
          [200, {}, JSON.generate("choices" => [{ "finish_reason" => "length", "message" => { "content" => '{"answer":0}' } }])]
        end
      end
    end
    connection = Faraday.new { |client| client.adapter :test, stubs }
    teacher = EasyAI::Distillation::Teachers::LocalHttp.new(endpoint: "http://127.0.0.1:8080/v1", model: "fixture",
      revision: "weights-sha", backend_revision: "test", connection: connection)

    assert_raises(ArgumentError) { teacher.call(EasyAI::Decision::DistillationAdapter.new.request(example)) }
    assert_equal 2, attempts
    assert_raises(ArgumentError) do
      EasyAI::Distillation::Teachers::LocalHttp.new(endpoint: "https://example.com/v1", model: "fixture", revision: "sha", backend_revision: "test")
    end
  end

  def test_teacher_content_can_be_exported_for_unlabeled_training_without_overwriting_gold
    Dir.mktmpdir do |dir|
      path = File.join(dir, "unlabeled.jsonl")
      File.write(path, JSON.generate(example.to_h.merge("target" => nil)) + "\n")
      data = EasyAI::Decision::Data::Dataset.new(path, require_target: false)
      artifact = collect(dir, data, Teacher.new)
      output = File.join(dir, "export")
      EasyAI::Decision::DistillationExport.write(dataset: data, artifact: artifact, output: output)
      exported = EasyAI::Decision::Data::Dataset.new(File.join(output, "train.jsonl"))

      assert_equal "r", exported[0].target
      assert_equal data[0].group_id, exported[0].group_id
      assert_equal artifact.fingerprint, JSON.parse(File.read(exported.path)).dig("label_provenance", "artifact_sha256")
      assert_raises(ArgumentError) { EasyAI::Decision::DistillationExport.write(dataset: data, artifact: artifact, output: output) }
      gold_artifact = collect(File.join(dir, "gold"), exported, Teacher.new)
      assert_raises(ArgumentError) do
        EasyAI::Decision::DistillationExport.write(dataset: exported, artifact: gold_artifact, output: File.join(dir, "bad"))
      end
    end
  end

  def test_collector_can_use_an_adapter_without_decision_fields
    Dir.mktmpdir do |dir|
      dataset = [{ "id" => "document", "text" => "A paragraph" }]
      dataset.define_singleton_method(:fingerprint) { "fixture-text-dataset" }
      adapter = Object.new
      adapter.define_singleton_method(:signature) { { "task" => "text_fixture", "version" => 1 } }
      adapter.define_singleton_method(:identity) { |row| { "id" => row.fetch("id"), "input_sha256" => EasyAI::Distillation::Fingerprint.call(row) } }
      adapter.define_singleton_method(:request) { |row| { "text" => row.fetch("text") } }
      adapter.define_singleton_method(:parse) { |_row, reply| { "generated_text" => reply.fetch("text") } }
      artifact = EasyAI::Distillation::Collector.new(teacher: Teacher.new, adapter: adapter, output: dir, purpose: "train", progress: nil).run(dataset)

      assert_equal "text_fixture", artifact.manifest.dig("adapter", "task")
      assert_equal '{"answer":0}', artifact.fetch("document").dig("supervision", "generated_text")
    end
  end

  def test_unlabeled_rows_cannot_silently_train_against_the_first_candidate
    Dir.mktmpdir do |dir|
      path = File.join(dir, "unlabeled.jsonl")
      File.write(path, JSON.generate(example.to_h.merge("target" => nil)) + "\n")
      data = EasyAI::Decision::Data::Dataset.new(path, require_target: false)

      assert_raises(ArgumentError) do
        EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(tiny_config),
          tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: data, output: File.join(dir, "train"))
      end
    end
  end

  def test_content_collection_export_and_training_commands_work_together
    Dir.mktmpdir do |dir|
      input = File.join(dir, "input.jsonl")
      File.write(input, JSON.generate(example.to_h.merge("target" => nil, "source" => "BoolQ")) + "\n")
      config = File.join(dir, "teacher.yml")
      File.write(config, "{}\n")
      artifact = File.join(dir, "teacher")
      io = StringIO.new
      status = EasyAI::Distillation::Cli.run(["--data", input, "--teacher-config", config, "--output", artifact,
        "--purpose", "train", "--prompt-profile", "task_specific"],
        out: io, err: io, teacher_factory: ->(**_config) { Teacher.new })

      assert_equal 0, status, io.string
      output = File.join(dir, "labels")

      assert_equal 0, EasyAI::Distillation::Cli.run(["--data", input, "--artifact", artifact, "--output", output],
        command: "distill-export", out: io, err: io), io.string
      File.write(File.join(dir, "student.yml"), tiny_config.to_h.to_yaml)
      EasyAI::Tokenizers::ByteBpe.new.save(File.join(dir, "tokenizer.json"))

      assert_equal 0, EasyAI::Decision::Cli.run(["train", "--data", File.join(output, "train.jsonl"),
        "--config", File.join(dir, "student.yml"), "--tokenizer", File.join(dir, "tokenizer.json"),
        "--output", File.join(dir, "student"), "--steps", "2"], out: io, err: io), io.string
      loaded = EasyAI::Decision::Checkpoint.load(File.join(dir, "student"))

      assert_equal 2, loaded[:metadata].dig("training", "step")
      assert_equal [example.group_id], loaded[:metadata].dig("training", "groups", "train")
    end
  end

  def test_soft_targets_reject_missing_candidates_and_invalid_probabilities
    adapter = EasyAI::Decision::DistillationAdapter.new
    reply = { "kind" => "candidate_probabilities", "temperature" => 1, "scoring_protocol" => "fixture", "probabilities" => { "r" => 1.0 } }

    assert_raises(ArgumentError) { adapter.parse(example, reply) }
    [{ "r" => -0.1, "b" => 1.1 }, { "r" => 0.4, "b" => 0.4 }, { "r" => Float::NAN, "b" => 0.5 }].each do |probabilities|
      assert_raises(ArgumentError) { adapter.parse(example, reply.merge("probabilities" => probabilities)) }
    end
    assert_raises(ArgumentError) do
      EasyAI::Distillation::Losses::SoftTargets.call(Torch.zeros([1, 2]), [[0.5, 0.5]], temperature: 0)
    end
  end

  private

  def collect(dir, data, teacher, purpose: "train")
    EasyAI::Distillation::Collector.new(teacher: teacher, adapter: EasyAI::Decision::DistillationAdapter.new,
      output: File.join(dir, purpose), purpose: purpose, progress: nil).run(data)
  end
end
