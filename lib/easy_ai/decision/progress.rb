module EasyAI
  module Decision
    class Progress
      def initialize(task:, total:, out: $stderr)
        @task, @total, @out = task, total, out
        @started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        @first_step = nil
        @interval = [total / 100, 1].max
      end

      def update(state, loss, device:)
        step = state.fetch("step")
        @first_step ||= step - 1
        return unless step == @first_step + 1 || step == @total || (step % @interval).zero?
        elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - @started
        completed = [step - @first_step, 1].max
        eta = elapsed / completed * [@total - step, 0].max
        filled = [[20 * step / @total, 0].max, 20].min
        bar = "=" * filled + "." * (20 - filled)
        row = state.fetch("history", []).last
        validation = row ? " val@#{row['step']}=#{format('%.4f', row['validation_loss'])}" : ""
        @out.puts("[#{@task}] [#{bar}] #{step}/#{@total} #{format('%5.1f', 100.0 * step / @total)}% " \
          "loss=#{format('%.4f', loss)}#{validation} device=#{device} elapsed=#{elapsed.round}s ETA~#{eta.round}s")
        @out.flush if @out.respond_to?(:flush)
      end
    end
  end
end
