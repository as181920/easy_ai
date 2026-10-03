module EasyAILearning
  module Training
    # Full-batch lessons intentionally avoid stochastic convergence as a correctness test.
    class Loop
      attr_reader :history, :optimizer, :model, :updates

      def initialize(model, kind: :adamw, lr: 0.02, decay: 0.0, clip: 5.0, schedule: false, seed: 1337)
        @model, @lr, @clip, @schedule, @seed = model, lr, clip, schedule, seed
        @optimizer = Optimizer.new(model.named_parameters, kind: kind, lr: lr, decay: decay)
        @history, @updates = [], 0
      end

      def state_dict
        { updates: updates, history: history, total_steps: @total_steps, lr: @lr, clip: @clip, schedule: @schedule, seed: @seed, optimizer: optimizer.state_dict }
      end

      def load_state_dict(saved)
        saved = saved.deep_symbolize_keys
        @updates, @history, @total_steps = saved.fetch(:updates), saved.fetch(:history), saved.fetch(:total_steps)
        @lr, @clip, @schedule, @seed = saved.fetch(:lr), saved.fetch(:clip), saved.fetch(:schedule), saved.fetch(:seed)
        optimizer.load_state_dict(saved.fetch(:optimizer))
        self
      end

      def run(steps:, validation: nil, total_steps: nil, stopping: nil, &objective)
        raise ArgumentError, "Positive steps required" unless steps > 0

        @total_steps = total_steps || updates + steps
        raise ArgumentError, "Invalid total step budget" unless @total_steps >= updates + steps
        steps.times do
          index = updates
          # Per-update seeds let full-batch lessons resume dropout/corruption exactly.
          Torch.manual_seed(@seed + index)
          model.train
          optimizer.zero_grad
          loss = objective.call
          raise FloatDomainError, "Nonfinite training loss" unless loss.item.finite?

          loss.backward
          grad = Math.clipped_gradients(model.parameters, @clip) if @clip
          lr = @schedule ? Math.learning_rate(index, total: @total_steps, base: @lr, warmup: [@total_steps / 10, @total_steps - 1].min) : @lr
          optimizer.param_groups.each { |group| group[:lr] = lr }
          optimizer.step
          @updates += 1
          row = { step: updates, train_loss: loss.detach.item, lr: lr, gradient: grad }
          if validation
            model.eval
            row[:validation_loss] = Torch.no_grad { validation.call.item }
          end
          history << row
          break if stopping && validation && stopping.observe(row.fetch(:validation_loss), model)
        end
        stopping.restore(model) if stopping && stopping.best_state
        model.eval
        self
      end
    end
  end
end
