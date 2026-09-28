module EasyAI
  module Decision
    module Growth
      class Controller
        attr_reader :state

        def initialize(config, state = nil)
          @config = config[:growth]
          @state = state || { "history" => [], "trials" => 0, "pending" => nil, "events" => [] }
        end

        def observe(train_loss:, validation_loss:, step:, model_config:)
          raise ArgumentError, "Nonfinite growth metric" unless train_loss.finite? && validation_loss.finite?
          if state["pending"]
            trial = state["pending"]
            trial["evaluations"] += 1
            if validation_loss < trial["baseline_loss"] - @config["min_delta"]
              state["events"] << { "step" => step, "action" => "accepted", "validation_loss" => validation_loss }
              state["pending"] = nil
              state["history"] = []
              return :keep
            end
            return :rollback if trial["evaluations"] >= @config["trial_evaluations"]
            return :wait
          end
          state["history"] << { "train" => train_loss, "validation" => validation_loss }
          state["history"] = state["history"].last(@config["patience"])
          return :wait unless @config["enabled"] && state["trials"] < @config["max_trials"] && capacity_available?(model_config)
          history = state["history"]
          return :wait if history.length < @config["patience"]
          # A plateau in both curves is only a heuristic for capacity: trial/rollback follows.
          plateau = %w[train validation].all? { |key| history.map { |h| h[key] }.minmax.then { |min, max| max - min <= @config["min_delta"] } }
          plateau ? :grow : :wait
        end

        def start_trial(path:, baseline_loss:, step:)
          state["trials"] += 1
          state["pending"] = { "checkpoint" => path, "baseline_loss" => baseline_loss, "evaluations" => 0 }
          state["events"] << { "step" => step, "action" => "trial_started", "operation" => @config["operation"] }
        end

        def reject(step:, reason: "no validation improvement")
          path = state.fetch("pending").fetch("checkpoint")
          state["events"] << { "step" => step, "action" => "rejected", "reason" => reason }
          state["pending"] = nil
          state["history"] = []
          path
        end

        private

        def capacity_available?(config)
          if @config["operation"] == "add_block"
            config[:model]["encoder_layers"] < @config["max_layers"]
          else
            widths = config[:model]["ffn_sizes"] || Array.new(config[:model]["encoder_layers"], config[:model]["ffn_size"])
            widths.min + @config["ffn_increment"] <= @config["max_ffn_size"]
          end
        end
      end
    end
  end
end
