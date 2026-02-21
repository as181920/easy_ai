#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "debug"
require "optparse"
require "unicode_plot"

ENV["LOG_LEVEL"] ||= "WARN"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai"

CUDA_ERROR = if defined?(Torch::CUDA::Error)
               Torch::CUDA::Error
             elsif defined?(Torch::Error)
               Torch::Error
             else
               StandardError
             end

DEFAULT_DATA_PATH = ENV.fetch("EASY_AI_DATA", "data/song.txt")

options = {
  data_path: DEFAULT_DATA_PATH,
  tokenizer: ENV.fetch("EASY_AI_TOKENIZER", "word"),
  num_merges: ENV.fetch("EASY_AI_MERGES", 2000).to_i,
  min_freq: ENV.fetch("EASY_AI_MIN_FREQ", 2).to_i,
  prompt: ENV.fetch("EASY_AI_PROMPT", "人间有味是清欢"),
  model: {
    block_size: ENV.fetch("EASY_AI_BLOCK_SIZE", 64).to_i,
    n_layer: ENV.fetch("EASY_AI_LAYERS", 2).to_i,
    n_head: ENV.fetch("EASY_AI_HEADS", 2).to_i,
    n_embd: ENV.fetch("EASY_AI_EMBED", 128).to_i,
    dropout: ENV.fetch("EASY_AI_DROPOUT", 0.1).to_f
  },
  training: {
    batch_size: ENV.fetch("EASY_AI_BATCH", 16).to_i,
    lr: ENV.fetch("EASY_AI_LR", 3e-4).to_f,
    weight_decay: ENV.fetch("EASY_AI_WEIGHT_DECAY", 0.01).to_f,
    max_iters: ENV.fetch("EASY_AI_ITERS", 200).to_i,
    log_interval: ENV.fetch("EASY_AI_LOG", 20).to_i,
    grad_clip: ENV.fetch("EASY_AI_GRAD_CLIP", 1.0).to_f,
    seed: ENV.fetch("EASY_AI_SEED", 1337).to_i
  },
  device_name: ENV.fetch("EASY_AI_DEVICE") { Torch::CUDA.available? ? "cuda" : "cpu" }
}

OptionParser.new do |opts|
  opts.banner = "Usage: bin/train_basic.rb [options]"

  opts.on("-d", "--data PATH", "Path to training text") { |v| options[:data_path] = v }
  opts.on("-t", "--tokenizer TYPE", "word or byte") { |v| options[:tokenizer] = v }
  opts.on("-m", "--merges N", Integer, "Number of BPE merges") { |v| options[:num_merges] = v }
  opts.on("-f", "--min-freq N", Integer, "Minimum pair frequency") { |v| options[:min_freq] = v }
  opts.on("-b", "--block-size N", Integer) { |v| options[:model][:block_size] = v }
  opts.on("-l", "--layers N", Integer) { |v| options[:model][:n_layer] = v }
  opts.on("-H", "--heads N", Integer) { |v| options[:model][:n_head] = v }
  opts.on("-e", "--embed N", Integer) { |v| options[:model][:n_embd] = v }
  opts.on("-B", "--batch N", Integer) { |v| options[:training][:batch_size] = v }
  opts.on("-i", "--iters N", Integer) { |v| options[:training][:max_iters] = v }
  opts.on("-p", "--prompt PROMPT", "Prompt for sampling") { |v| options[:prompt] = v }
  opts.on("--device NAME", "Device to train on (cpu/cuda)") { |v| options[:device_name] = v }
end.parse!

def read_text_file(path)
  File.read(path, mode: "rb").encode("UTF-8", invalid: :replace, undef: :replace, replace: "")
end

def locate_corpus(path)
  if File.directory?(path)
    files = Dir.glob(File.join(path, "**", "*.txt")).sort
    raise "No .txt files found under #{path}" if files.empty?

    text = files.map { |f| read_text_file(f) }.join("\n\n")
    { text: text, label: "#{path} (#{files.count} *.txt files)", file_count: files.count }
  elsif File.file?(path)
    { text: read_text_file(path), label: path, file_count: 1 }
  else
    raise "Path not found: #{path}"
  end
end

def resolve_device(name)
  device = Torch.device(name)

  if device.type == :cuda && !Torch::CUDA.available?
    warn "CUDA not available; using cpu"
    return Torch.device("cpu")
  end

  begin
    Torch.zeros([1], device: device)
    device
  rescue CUDA_ERROR
    warn "CUDA initialization failed; using cpu"
    Torch.device("cpu")
  end
rescue StandardError
  warn "Unknown device #{name}; using cpu"
  Torch.device("cpu")
end

def train_with_device(device, dataset, tokenizer, model_opts, training_opts)
  training_cfg = training_opts.merge(device: device)
  config = EasyAI::Config.new(model: model_opts, training: training_cfg)

  Torch.manual_seed(config.training[:seed]) if config.training[:seed]

  model = EasyAI::Models::GPT.new(config.model)
  model.to(device)

  batcher = EasyAI::Data::Batcher.new(
    dataset: dataset,
    batch_size: config.training[:batch_size],
    device: device,
    seed: config.training[:seed]
  )

  trainer = EasyAI::Trainers::Trainer.new(
    model: model,
    batcher: batcher,
    config: config.training
  )

  trainer.train

  [trainer, model, config]
end

preferred_device = resolve_device(options[:device_name])
corpus_info = locate_corpus(options[:data_path])
text = corpus_info[:text]

tokenizer_class = case options[:tokenizer]
                  when "byte" then EasyAI::Tokenizers::ByteBpe
                  else EasyAI::Tokenizers::WordBpe
                  end

tokenizer = tokenizer_class.new(num_merges: options[:num_merges], min_freq: options[:min_freq])

dataset = EasyAI::Data::TextDataset.new(
  tokenizer: tokenizer,
  text: text,
  block_size: options[:model][:block_size]
)
model_config = options[:model].merge(vocab_size: tokenizer.vocab_size)
training_config = options[:training]

puts "Training on #{corpus_info[:label]} (#{text.length} chars) with #{tokenizer.vocab_size} tokens"

trainer = nil
model = nil
config = nil
device_used = preferred_device

begin
  trainer, model, config = train_with_device(preferred_device, dataset, tokenizer, model_config, training_config)
rescue CUDA_ERROR => e
  warn "CUDA error during training (#{e.message}); retrying on cpu"
  device_used = Torch.device("cpu")
  trainer, model, config = train_with_device(device_used, dataset, tokenizer, model_config, training_config)
end

loss_history = trainer.loss_history

plot = UnicodePlot.lineplot(
  (0...loss_history.length).to_a,
  loss_history,
  height: 12,
  width: 70,
  title: "Training Loss",
  xlabel: "Iteration",
  ylabel: "Loss"
)

puts plot

prompt_ids = tokenizer.encode(options[:prompt])
prompt_ids = prompt_ids.last(config.model[:block_size])
input_tensor = Torch.tensor([prompt_ids], dtype: :int64, device: device_used)

generated = model.generate(input_tensor, max_new_tokens: 64)
output_ids = generated.cpu[0].to_a
puts "\nSample:"
puts tokenizer.decode(output_ids)
