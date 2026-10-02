require_relative "fitting"
loaded = EasyAI::Decision::Checkpoint.load(EvidenceExperiment.parent(1337))
config = loaded.fetch(:model).config.with(model: { position_encoding: "rotary", evidence_head: false }, training: { early_stopping_patience: 0 })
model = EasyAI::Decision::ChoiceModel.new(config)
model.load_state_dict(loaded.fetch(:model).state_dict)
model.to("cuda")
optimizer = EasyAI::Optim::AdamW.new(model.named_parameters, learning_rate: 0.0001)
policy = EasyAI::Runtime::DevicePolicy.new(requested: "cuda", budget_mib: 4096)
batch_size = Integer(ARGV.fetch(0, "4"))
Torch.manual_seed(1337)
batch = {
  state_ids: Torch.randint(5, 12000, [batch_size, 256], dtype: :int64, device: "cuda"),
  state_mask: Torch.ones([batch_size, 256], dtype: :bool, device: "cuda"),
  option_ids: Torch.randint(5, 12000, [batch_size, 18, 128], dtype: :int64, device: "cuda"),
  option_mask: Torch.ones([batch_size, 18, 128], dtype: :bool, device: "cuda"),
  candidate_mask: Torch.ones([batch_size, 18], dtype: :bool, device: "cuda"),
  targets: Torch.zeros([batch_size], dtype: :int64, device: "cuda")
}
samples = []
4.times do |i|
  model.train
  optimizer.zero_grad
  loss = Torch::NN::Functional.cross_entropy(model.call(batch), batch.fetch(:targets))
  loss.backward
  samples << { "phase" => "backward", "round" => i, "mib" => policy.check_budget!("cuda") }
  optimizer.step
  loss = nil
  GC.start
  model.eval
  value = Torch.no_grad { model.call(batch).sum.item }
  GC.start
  samples << { "phase" => "validation", "round" => i, "mib" => policy.check_budget!("cuda"), "finite" => value.finite? }
end
FileUtils.mkdir_p(File.join(EvidenceExperiment::ROOT, "tmp"))
SemanticCoverage.write_json(File.join(EvidenceExperiment::ROOT, "tmp/natural-memory-profile-#{batch_size}.json"), { "batch_size" => batch_size, "samples" => samples,
  "scope" => "Dense pilot shape: 18 candidates, state256, candidate128; boundary process memory, not allocator peak or arbitrary API candidate counts. Torch.rb exposes no CUDA allocator statistics." })
puts JSON.generate(samples)
