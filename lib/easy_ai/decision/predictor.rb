require "digest"
require "thread"

module EasyAI
  module Decision
    class Predictor
      attr_reader :model, :tokenizer, :device, :calibrator, :calibrated, :cache_hits

      def self.load(path, device: "auto", candidate_chunk_size: nil)
        checkpoint = Checkpoint.load(path)
        calibration = checkpoint[:metadata]["calibration"]
        new(model: checkpoint[:model], tokenizer: checkpoint[:tokenizer], device: device,
            temperature: calibration ? calibration.fetch("temperature") : 1.0,
            calibrated: !calibration.nil?, candidate_chunk_size: candidate_chunk_size)
      end

      def initialize(model:, tokenizer:, device: "auto", temperature: 1.0, calibrated: false, candidate_chunk_size: nil)
        @model, @tokenizer, @calibrated = model, tokenizer, calibrated
        @policy = Runtime::DevicePolicy.new(requested: device, budget_mib: model.config[:runtime]["gpu_memory_budget_mib"])
        @device = @policy.resolve
        @calibrator = Calibrator.new(temperature: temperature)
        @collator = Data::Collator.new(tokenizer: tokenizer, config: model.config)
        @chunk_size = candidate_chunk_size || model.config[:runtime]["candidate_chunk_size"]
        raise ArgumentError, "candidate_chunk_size must be positive" unless @chunk_size.is_a?(Integer) && @chunk_size > 0
        @cache, @cache_hits, @mutex = {}, 0, Mutex.new
        move_model!
      end

      def probabilities(state:, question:, options:)
        example = Data::Example.new({ state: state, question: question, options: options }, require_target: false)
        @mutex.synchronize do
          before = @collator.truncated
          logits = score_with_fallback(example)
          probs = calibrator.probabilities(logits)
          { "probabilities" => example.options.map { |option| option.fetch("id") }.zip(probs).to_h,
           "calibrated" => calibrated, "temperature" => calibrator.temperature,
           "device" => device, "truncated_segments" => @collator.truncated - before }
        end
      end

      def logits(example)
        @mutex.synchronize { score_with_fallback(example) }
      end

      def clear_cache
        @cache.clear
      end

      private

      def move_model!
        model.to(device) unless model.parameters.first.device.type.to_s == device
        model.eval
      rescue Torch::Error => error
        raise unless device == "cuda" && @policy.recoverable?(error)
        @device = "cpu"
        model.to(device).eval
      end

      def score_with_fallback(example)
        score(example)
      rescue Torch::Error, Runtime::DevicePolicy::MemoryBudgetExceeded => error
        raise unless device == "cuda" && @policy.recoverable?(error)
        clear_cache
        if @chunk_size > 1
          @chunk_size = [@chunk_size / 2, 1].max
        else
          @device = "cpu"
          move_model!
        end
        GC.start
        retry
      ensure
        # Public API callers may keep one predictor alive indefinitely.
        # Only the bounded state cache should retain tensors between requests.
        GC.start
      end

      def score(example)
        model.eval
        if model.config[:model]["encoding_mode"] == "joint"
          return Torch.no_grad do
            example.options.each_slice(@chunk_size).flat_map do |slice|
              values = score_chunk(example, slice, nil, nil)
              GC.start
              @policy.check_budget!(device)
              values
            end
          end
        end
        Torch.no_grad do
          ids = @collator.state_tokens(example.state)
          key = Digest::SHA256.hexdigest(ids.pack("L<*"))
          entry = @cache.delete(key)
          if entry
            @cache_hits += 1
          else
            tensor = Torch.tensor([ids], dtype: :int64, device: device)
            mask = Torch.ones([1, ids.length], dtype: :bool, device: device)
            entry = [model.encode_state(tensor, mask), mask]
          end
          @cache[key] = entry
          @cache.shift while @cache.length > model.config[:runtime]["cache_entries"]
          memory, memory_mask = entry
          scores = example.options.each_slice(@chunk_size).flat_map do |slice|
            values = score_chunk(example, slice, memory, memory_mask)
            GC.start
            @policy.check_budget!(device)
            values
          end
          scores
        end
      end

      def score_chunk(example, slice, memory, memory_mask)
        # Collator requires >=2 public options; duplicate then ignore singleton padding.
        padded = slice.length == 1 ? slice + [{ "id" => "__padding__#{slice.first['id']}", "text" => slice.first.fetch("text") }] : slice
        part = Data::Example.new({ state: example.state, question: example.question, options: padded }, require_target: false)
        batch = @collator.call([part], device: device)
        output = if model.config[:model]["encoding_mode"] == "joint"
          model.call(batch)
                 else
          model.score_candidates(memory, memory_mask, batch[:option_ids], batch[:option_mask], batch[:candidate_mask],
            answer_mask: batch[:answer_mask])
                 end
        output.cpu.to_a.first.first(slice.length)
      end
    end
  end
end
