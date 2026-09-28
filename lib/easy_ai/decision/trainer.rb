require "json"
require "fileutils"

module EasyAI
  module Decision
    class Trainer
      attr_reader :model, :tokenizer, :optimizer, :state, :device, :last_checkpoint

      def initialize(model:, tokenizer:, dataset:, output:, validation: nil, task: :choice,
                     restored: nil, device: nil, logger: EasyAI.logger)
        @model, @tokenizer, @dataset, @output, @validation, @task, @logger = model, tokenizer, dataset, output, validation, task.to_sym, logger
        raise ArgumentError, "Unknown training task" unless %i[choice mlm].include?(@task)
        raise ArgumentError, "Dataset task mismatch" unless @dataset.kind == @task
        raise ArgumentError, "Tokenizer exceeds model vocabulary" if tokenizer.vocab_size > model.config[:model]["vocab_size"]
        @config = model.config
        raise ArgumentError, "Automatic growth needs validation data" if @config[:growth]["enabled"] && !validation
        raise ArgumentError, "Early stopping needs validation data" if @config[:training]["early_stopping_patience"] > 0 && !validation
        Data::Dataset.assert_disjoint!(dataset, validation)
        @state = restored ? deep_copy(restored.fetch("training")) : { "step" => 0, "examples_seen" => 0, "architecture_version" => 1,
          "history" => [], "task" => @task.to_s, "datasets" => {}, "groups" => { "train" => [], "validation" => [] } }
        raise ArgumentError, "Resume task mismatch; use init for transfer" unless state.fetch("task") == @task.to_s
        expected = state.fetch("datasets", {})["train"]
        raise ArgumentError, "Resume dataset fingerprint mismatch" if expected && expected != dataset.fingerprint
        if state.fetch("datasets", {}).key?("validation") && state["datasets"]["validation"] != validation&.fingerprint
          raise ArgumentError, "Resume validation fingerprint mismatch"
        end
        state["datasets"] = { "train" => dataset.fingerprint, "validation" => validation&.fingerprint }
        state["groups"] ||= { "train" => [], "validation" => [] }
        state["groups"]["train"] |= dataset.groups.to_a
        state["groups"]["validation"] |= validation.groups.to_a if validation
        if (state["groups"]["train"] & state["groups"]["validation"]).any?
          raise ArgumentError, "Training and validation groups overlap (including earlier stages)"
        end
        @policy = Runtime::DevicePolicy.new(requested: device || @config[:training]["device"], budget_mib: @config[:runtime]["gpu_memory_budget_mib"], logger: logger)
        @device = "cpu"
        @optimizer = make_optimizer
        @optimizer.load_state_dict(restored["optimizer"]) if restored && restored["optimizer"]
        @microbatch = state.fetch("microbatch", @config[:training][@task == :choice ? "choice_microbatch" : "mlm_microbatch"])
        @accumulation = state.fetch("gradient_accumulation", @config[:training]["gradient_accumulation"])
        @growth = Growth::Controller.new(@config, state["growth"])
        @collator = Data::Collator.new(tokenizer: tokenizer, config: @config)
        @masking = Data::Masking.new(tokenizer: tokenizer, config: @config)
        @language_indexes = Hash.new { |h, key| h[key] = [] }
        @source_indexes = Hash.new { |sources, source| sources[source] = Hash.new { |languages, language| languages[language] = [] } }
        @label_indexes = Hash.new { |hash, language| hash[language] = Hash.new { |labels, label| labels[label] = [] } }
        dataset.each_with_index do |example, index|
          language = example.is_a?(Data::Example) ? example.language : example.fetch("language", "und")
          @language_indexes[language] << index
          source = example.is_a?(Data::Example) ? example.source : example.fetch("source", "local")
          @source_indexes[source][language] << index
          @label_indexes[language][[example.question, example.target]] << index if @task == :choice
        end
        @candidate_sampler = Data::CandidateSampler.new(dataset) if @task == :choice && @config[:training]["resample_negatives"]
        @pair_sampler = Data::PairSampler.new(dataset) if @task == :choice && @config[:training]["paired_sampling"]
        FileUtils.mkdir_p(output)
        save_checkpoint # CPU snapshot exists before trying GPU allocations.
        transfer_to(@policy.resolve)
      end

      def self.resume(checkpoint, dataset:, output:, validation: nil, steps: nil, device: nil, **kwargs)
        loaded = Checkpoint.load(checkpoint)
        if loaded[:metadata].dig("training", "stop_reason")
          loaded[:metadata]["training"].delete("stop_reason")
          loaded[:metadata]["training"]["bad_evaluations"] = 0
        end
        model = loaded[:model]
        if steps
          config = model.config.with(training: { steps: steps })
          replacement = ChoiceModel.new(config)
          replacement.load_state_dict(model.state_dict)
          model = replacement
        end
        new(model: model, tokenizer: loaded[:tokenizer], dataset: dataset, output: output,
            validation: validation, task: loaded[:metadata]["training"].fetch("task").to_sym,
            restored: loaded[:metadata], device: device, **kwargs)
      end

      def train(steps: @config[:training]["steps"])
        while state["step"] < steps
          begin
            loss = update_step
            state["step"] += 1
            state["examples_seen"] += @microbatch * @accumulation
            state["last_train_loss"] = loss
            trace = { "step" => state["step"], "train_loss" => loss, "device" => device,
                      "learning_rate" => optimizer.learning_rate, "architecture_version" => state["architecture_version"] }
            File.open(File.join(@output, "training.jsonl"), "a") { |file| file.puts(JSON.generate(trace)) }
            @logger.info("Decision #{@task} step=#{state['step']} loss=#{loss.round(6)} device=#{device}")
            yield(state, loss) if block_given?
            validation_due = @validation && (state["step"] % @config[:training]["eval_every"]).zero?
            save_checkpoint if (state["step"] % @config[:training]["checkpoint_every"]).zero? || (device == "cuda" && validation_due)
            validate_and_grow(loss) if validation_due
            if state["stop_reason"] && !@growth.state["pending"]
              @logger.info("Decision #{@task} stopped: #{state['stop_reason']}")
              break
            end
          rescue Torch::Error, Runtime::DevicePolicy::MemoryBudgetExceeded => error
            raise unless device == "cuda" && @policy.recoverable?(error)
            recover_from_gpu(error)
          end
          GC.start
        end
        # An unevaluated growth trial never silently becomes the final model.
        rollback_growth("training budget ended before trial acceptance") if @growth.state["pending"]
        save_checkpoint
      end

      def save_checkpoint
        state["microbatch"], state["gradient_accumulation"] = @microbatch, @accumulation
        state["growth"] = deep_copy(@growth.state) if @growth
        @last_checkpoint = Checkpoint.save(@output, model: model, tokenizer: tokenizer, optimizer: optimizer, training_state: state)
      end

      private

      def make_optimizer
        Optim::AdamW.new(model.named_parameters, learning_rate: @config[:training]["learning_rate"], weight_decay: @config[:training]["weight_decay"])
      end

      def transfer_to(destination)
        saved_optimizer = optimizer.state_dict
        model.to(destination)
        @device = destination
        @optimizer = make_optimizer
        optimizer.load_state_dict(saved_optimizer)
        @policy.check_budget!(device)
      rescue Torch::Error, Runtime::DevicePolicy::MemoryBudgetExceeded => error
        raise unless destination == "cuda" && @policy.recoverable?(error)
        @logger.warn("GPU model placement failed; restoring CPU checkpoint: #{error.message.lines.first}")
        restore_checkpoint(last_checkpoint, destination: "cpu")
      end

      def update_step
        model.train
        optimizer.zero_grad
        step = state["step"]
        warmup = @config[:training]["warmup_steps"]
        scale = warmup.zero? ? 1.0 : [(step + 1).to_f / warmup, 1.0].min
        scale *= 0.25 if state.fetch("growth_warmup_until", 0) > step
        optimizer.learning_rate = @config[:training]["learning_rate"] * scale
        rng = Random.new(@config[:training]["seed"] + step * 1009)
        languages = @language_indexes.keys.sort
        all_indices = Array.new(@microbatch * @accumulation) do
          if @config[:training]["balance_sources"]
            source = @source_indexes.keys.sort.sample(random: rng)
            language = @source_indexes.fetch(source).keys.sort.sample(random: rng)
            next @source_indexes.fetch(source).fetch(language).sample(random: rng)
          end
          language = languages.sample(random: rng)
          indexes = if @task == :choice && @config[:training]["balance_labels"]
            @label_indexes.fetch(language).values.sample(random: rng)
                    else
            @language_indexes.fetch(language)
                    end
          indexes.sample(random: rng)
        end
        all_indices = @pair_sampler.sample(@microbatch * @accumulation, rng: rng) if @pair_sampler
        total_loss = 0.0
        all_indices.each_slice(@microbatch).with_index do |indices, micro|
          seed = (@config[:training]["seed"] + step * 1009 + micro) % (2**31)
          Torch.manual_seed(seed)
          Torch::CUDA.manual_seed_all(seed) if device == "cuda"
          examples = indices.map { |index| @dataset[index] }
          if @candidate_sampler
            candidate_rng = Random.new(seed ^ 0x5EED)
            examples = examples.map { |example| @candidate_sampler.call(example, rng: candidate_rng) }
          end
          loss = loss_for(examples, seed: seed)
          number = loss.item
          raise FloatDomainError, "Non-finite training loss" unless number.finite?
          @policy.check_budget!(device) if micro.zero?
          (loss / @accumulation).backward
          total_loss += number / @accumulation
          loss = nil
          GC.start
        end
        optimizer.clip_grad_norm!(@config[:training]["grad_clip"])
        optimizer.step
        @policy.check_budget!(device)
        total_loss
      end

      def loss_for(examples, seed:)
        if @task == :choice
          batch = @collator.call(examples, device: device)
          Torch::NN::Functional.cross_entropy(model.call(batch), batch[:targets])
        else
          batch = @masking.call(examples, seed: seed, device: device)
          logits = model.mlm_logits(batch[:ids], mask: batch[:mask], positions: batch[:positions])
          Torch::NN::Functional.cross_entropy(logits, batch[:targets])
        end
      end

      def validation_loss
        model.eval
        total, count = 0.0, 0
        GC.start
        Torch.no_grad do
          @validation.each_slice(@microbatch).with_index do |rows, index|
            value, weight = validation_batch(rows, index)
            total += value * weight
            count += weight
            # The method has returned: no batch tensors escape into this scope.
            # Ruby GC cannot see native tensor bytes; no_grad does not free them.
            GC.start
          end
        end
        total / count
      ensure
        GC.start
      end

      def validation_batch(rows, index)
        return [loss_for(rows, seed: 0).item, rows.length] if @task == :choice
        # Fixed masks across evaluations; weight MLM loss by masked tokens.
        batch = @masking.call(rows, seed: @config[:training]["seed"] + index, device: device)
        logits = model.mlm_logits(batch[:ids], mask: batch[:mask], positions: batch[:positions])
        [Torch::NN::Functional.cross_entropy(logits, batch[:targets]).item, batch[:targets].shape[0]]
      end

      def validate_and_grow(train_loss)
        memory_before = @policy.process_memory_mib if device == "cuda"
        value = validation_loss
        row = { "step" => state["step"], "train_loss" => train_loss, "validation_loss" => value,
               "device" => device, "truncated_segments" => @collator.truncated,
               "gpu_process_mib_before_validation" => memory_before,
               "gpu_process_mib_after_validation" => device == "cuda" ? @policy.process_memory_mib : nil }
        state["history"] << row
        @logger.info("Decision #{@task} validation step=#{state['step']} loss=#{value.round(6)} device=#{device}")
        File.open(File.join(@output, "metrics.jsonl"), "a") { |file| file.puts(JSON.generate(row)) }
        action = @growth.observe(train_loss: train_loss, validation_loss: value, step: state["step"], model_config: @config)
        record_selection(value) unless @growth.state["pending"]
        if action == :grow
          start_growth(value)
        elsif action == :rollback
          rollback_growth("no validation improvement")
        end
      end

      def record_selection(value)
        previous = state["best_validation_loss"]
        significant = previous.nil? || value < previous - @config[:training]["selection_min_delta"]
        state["bad_evaluations"] = significant ? 0 : state.fetch("bad_evaluations", 0) + 1
        if previous.nil? || value < previous
          state["best_validation_loss"] = value
          state["best_step"] = state["step"]
          # Selection artifacts are inference/transfer weights. latest keeps resumable optimizer state.
          state["best_checkpoint"] = File.expand_path(Checkpoint.save(File.join(@output, "best"),
            model: model, tokenizer: tokenizer, training_state: state))
        end
        patience = @config[:training]["early_stopping_patience"]
        if patience > 0 && state["bad_evaluations"] >= patience
          state["stop_reason"] = "validation did not improve for #{patience} evaluations"
        end
      end

      def start_growth(baseline)
        parent = save_checkpoint
        saved_optimizer = optimizer.state_dict
        grown = if @config[:growth]["operation"] == "add_block"
          Growth::AddBlock.apply(model)
                else
          widths = @config[:model]["ffn_sizes"] || Array.new(@config[:model]["encoder_layers"], @config[:model]["ffn_size"])
          layer = widths.each_index.min_by { |i| widths[i] }
          Growth::WidenFfn.apply(model, layer: layer, size: widths[layer] + @config[:growth]["ffn_increment"])
                end
        @growth.start_trial(path: parent, baseline_loss: baseline, step: state["step"])
        @model, @config = grown, grown.config
        @model.to(device)
        @optimizer = make_optimizer
        optimizer.load_state_dict(saved_optimizer, allow_growth: true)
        state["architecture_version"] += 1
        state["growth_warmup_until"] = state["step"] + @config[:training]["eval_every"]
        @policy.check_budget!(device)
        save_checkpoint
      end

      def rollback_growth(reason)
        current_step = state["step"]
        examples_seen = state["examples_seen"]
        parent = @growth.reject(step: current_step, reason: reason)
        growth_state = deep_copy(@growth.state)
        restore_checkpoint(parent, destination: device)
        @growth = Growth::Controller.new(@config, growth_state)
        # Trial compute consumes the overall budget; do not silently repeat it.
        state["step"] = current_step
        state["examples_seen"] = examples_seen
        state["growth"] = growth_state
      end

      def recover_from_gpu(error)
        @logger.warn("GPU capacity failure; restore committed checkpoint: #{error.message.lines.first}")
        if @growth.state["pending"]
          rollback_growth("GPU memory budget")
          save_checkpoint
          return
        end
        effective = @microbatch * @accumulation
        next_batch = @microbatch > 1 && @microbatch.even? ? @microbatch / 2 : 1
        destination = @microbatch > 1 ? "cuda" : "cpu"
        restore_checkpoint(last_checkpoint, destination: "cpu")
        @microbatch, @accumulation = next_batch, effective / next_batch
        GC.start
        transfer_to(destination)
      end

      def restore_checkpoint(path, destination:)
        loaded = Checkpoint.load(path)
        @model, @config = loaded[:model], loaded[:model].config
        @state = loaded[:metadata].fetch("training")
        @growth = Growth::Controller.new(@config, state["growth"])
        @device = destination
        @model.to(destination)
        @optimizer = make_optimizer
        optimizer.load_state_dict(loaded[:metadata]["optimizer"])
      end

      def deep_copy(value)
        JSON.parse(JSON.generate(value))
      end
    end
  end
end
