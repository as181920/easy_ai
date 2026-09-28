#!/usr/bin/env ruby
require "bundler/setup"
require "tmpdir"
require "json"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
require "easy_ai"

abort "CUDA is not accessible; run in the local GPU environment" unless Torch::CUDA.available?
config = EasyAI::Decision::Config.new(model: { vocab_size: 262, hidden_size: 32, encoder_layers: 2,
  attention_heads: 4, ffn_size: 64, dropout: 0.0 },
  training: { device: "cuda", steps: 2, choice_microbatch: 1, gradient_accumulation: 1, checkpoint_every: 1 })
tokenizer = EasyAI::Tokenizers::ByteBpe.new
example = EasyAI::Decision::Data::Example.new({ id: "check", state: "颜色是红色", question: "Color?",
  options: [{ id: "red", text: "红色 red" }, { id: "blue", text: "蓝色 blue" }], target: "red" })
Torch.manual_seed(1)
model = EasyAI::Decision::ChoiceModel.new(config).eval
collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config)
cpu = Torch.no_grad { model.call(collator.call([example])).detach.clone }
model.to("cuda")
gpu = Torch.no_grad { model.call(collator.call([example], device: "cuda")).cpu }
difference = (cpu - gpu).abs.max.item
raise "CPU/CUDA logit difference #{difference}" unless difference < 1e-4

Dir.mktmpdir("easy-ai-gpu-verification") do |directory|
  path = File.join(directory, "data.jsonl")
  File.write(path, JSON.generate(example.to_h) + "\n")
  dataset = EasyAI::Decision::Data::Dataset.new(path)
  trainer = EasyAI::Decision::Trainer.new(model: model.cpu, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "gpu"))
  trainer.train
  raise "GPU training unexpectedly fell back" unless trainer.device == "cuda"
  resumed = EasyAI::Decision::Trainer.resume(trainer.last_checkpoint, dataset: dataset,
    output: File.join(directory, "cpu"), steps: 3, device: "cpu")
  resumed.train
  raise "CPU resume failed" unless resumed.device == "cpu" && resumed.state["step"] == 3
  budget_config = config.with(runtime: { gpu_memory_budget_mib: 1 })
  fallback = EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(budget_config), tokenizer: tokenizer,
    dataset: dataset, output: File.join(directory, "fallback"))
  fallback.train
  raise "Memory budget did not trigger CPU fallback" unless fallback.device == "cpu"
  puts JSON.pretty_generate("cpu_cuda_max_logit_difference" => difference, "gpu_updates" => trainer.state["step"],
    "resumed_on_cpu_step" => resumed.state["step"], "forced_budget_fallback" => fallback.device)
end
