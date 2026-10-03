require "optparse"

module EasyAILearning
  module Course
    module Predict
      module_function

      DEFAULTS = {
        "02_training" => "adamw", "03_diagnostics" => "baseline", "04_autoencoder" => "bottleneck", "05_cnn" => "cnn",
        "06_resnet" => "residual", "08_rnn" => "gru", "09_seq2seq" => "seq2seq", "10_attention" => "cross-attention",
        "11_transformer" => "encoder-decoder", "12_gpt" => "gpt", "13_generative" => "vae", "14_transfer_learning" => "full",
        "15_rl" => "ppo", "16_capstone" => "cnn-1337"
      }.freeze

      def build(saved)
        config = saved.fetch("config").deep_symbolize_keys
        config[:kind] = config[:kind].to_sym if config[:kind]
        # A whitelist avoids loading arbitrary classes from a supplied JSON artifact.
        case saved.fetch("class")
        when "Torch::NN::Embedding" then Torch::NN::Embedding.new(config.fetch(:num_embeddings), config.fetch(:embedding_dim))
        when "EasyAILearning::BasicNN::Mlp" then BasicNN::Mlp.new(**config)
        when "EasyAILearning::Autoencoder::Model" then Autoencoder::Model.new(**config)
        when "EasyAILearning::Autoencoder::Convolutional" then Autoencoder::Convolutional.new
        when "EasyAILearning::CNN::Model" then CNN::Model.new(**config)
        when "EasyAILearning::Resnet::Model" then Resnet::Model.new(**config)
        when "EasyAILearning::RNN::Model" then RNN::Model.new(**config)
        when "EasyAILearning::Seq2seq::Model" then Seq2seq::Model.new(**config)
        when "EasyAILearning::Transformer::SequenceModel" then Transformer::SequenceModel.new(**config)
        when "EasyAILearning::Transformer::EncoderDecoder" then Transformer::EncoderDecoder.new(**config)
        when "EasyAILearning::GPT::Model" then GPT::Model.new(config)
        when "EasyAILearning::Generative::Vae" then Generative::Vae.new(**config)
        when "EasyAILearning::Generative::Gan" then Generative::Gan.new
        when "EasyAILearning::Generative::Diffusion" then Generative::Diffusion.new(**config)
        when "EasyAILearning::Generative::MaskedAutoencoder" then Generative::MaskedAutoencoder.new(**config)
        when "EasyAILearning::Transfer::AdaptedMlp" then Transfer::AdaptedMlp.new(**config)
        when "EasyAILearning::RL::ActorCritic" then RL::ActorCritic.new(**config)
        else raise ArgumentError, "Unsupported model class"
        end
      end

      def sample(model)
        case model
        when Torch::NN::Embedding then { input: [[1, 2, 1]] }
        when BasicNN::Mlp then { input: [Array.new(model.hidden.weight.shape[1], 0.25)] }
        when CNN::Model, Resnet::Model, Autoencoder::Convolutional, Generative::MaskedAutoencoder
          { input: Data.images(count: 1).first }
        when RNN::Model, GPT::Model, Transformer::SequenceModel then { input: [[0, 1, 2]] }
        when Seq2seq::Model, Transformer::EncoderDecoder then { input: [[3, 4, 5, 6]] }
        when Generative::Gan, Generative::Diffusion then { input: [[0.1, -0.1]], time: [0] }
        when RL::ActorCritic then { input: [[1, 0, 0, 0, 0]] }
        else { input: [[1.0, 0.0, 0.5, 0.0]] }
        end
      end

      def run(chapter)
        options = { device: "auto", model: "runs/learning/#{chapter}/default/#{DEFAULTS.fetch(chapter)}-model.json" }
        OptionParser.new do |parser|
          parser.banner = "Usage: ruby learning/#{chapter}/predict.rb [--model PATH] [--input JSON_FILE]"
          parser.on("--model PATH") { |v| options[:model] = v }
          parser.on("--device NAME") { |v| options[:device] = v }
          parser.on("--input PATH") { |v| options[:input] = v }
        end.parse!
        saved = JSON.parse(File.read(options[:model]))
        device = EasyAI::Runtime::DevicePolicy.new(requested: options[:device]).resolve
        model = build(saved).to(device)
        Artifacts.load_model(options[:model], model)
        payload = options[:input] ? JSON.parse(File.read(options[:input])).deep_symbolize_keys : sample(model)
        integer = [Torch::NN::Embedding, RNN::Model, Seq2seq::Model, Transformer::SequenceModel, Transformer::EncoderDecoder, GPT::Model].any? { |klass| model.is_a?(klass) }
        input = Data.tensor(payload.fetch(:input), device: device, integer: integer)
        result = Torch.no_grad do
          case model
          when Seq2seq::Model, Transformer::EncoderDecoder then model.generate(input)
          when GPT::Model then model.generate(input, max_new_tokens: 8, top_k: 1)
          when Generative::Diffusion then model.call(input, Data.tensor(payload.fetch(:time), device: device, integer: true))
          when Generative::MaskedAutoencoder then model.call(input, visible: payload.fetch(:visible, (0...8).to_a))
          else model.call(input)
          end
        end
        result = result.map { |t| t.cpu.to_a } if result.is_a?(Array)
        result = result.cpu.to_a if result.is_a?(Torch::Tensor)
        puts JSON.pretty_generate(model: saved.fetch("class"), input: payload, output: result)
      end
    end
  end
end
