require "json"
require "fileutils"
require "open3"
require "cgi"
require "unicode_plot"

module EasyAI
  module Decision
    class TrainingReport
      def self.check_plotter!
        _, error, status = Open3.capture3("gnuplot", "--version")
        raise ArgumentError, "gnuplot is unavailable: #{error}" unless status.success?
      rescue Errno::ENOENT
        raise ArgumentError, "Training charts require gnuplot. Install gnuplot, then retry."
      end

      def initialize(root, out: $stderr)
        @root, @out = File.expand_path(root), out
        @directory = File.join(@root, "report")
      end

      def write
        self.class.check_plotter!
        raise ArgumentError, "Run directory does not exist: #{@root}" unless File.directory?(@root)
        @series = %w[mlm choice].to_h do |stage|
          [stage, { "train" => trace(stage, "training.jsonl", "train_loss"),
                    "validation" => trace(stage, "metrics.jsonl", "validation_loss") }]
        end
        @memory = %w[mlm choice].to_h do |stage|
          [stage, %w[before after].to_h { |point| [point, trace(stage, "metrics.jsonl", "gpu_process_mib_#{point}_validation", optional: true)] }]
        end.reject { |_stage, series| series.values.all?(&:empty?) }
        raise ArgumentError, "No training traces in #{@root}" if @series.values.all? { |series| series["train"].empty? }
        @calibration = read_json("stage-results/calibration.json")&.fetch("calibration", nil)
        @test = read_json("stage-results/test.json")
        @diagnostic = read_json("stage-results/diagnostic.json")
        @semantic_diagnostic = read_json("stage-results/semantic-diagnostic.json")
        @summary = read_json("summary.json") || {}
        FileUtils.mkdir_p(@directory)
        write_series
        write_console
        render("loss", loss_script)
        render("evaluation", evaluation_script) if @calibration && @test
        render("memory", memory_script) unless @memory.empty?
        File.write(File.join(@directory, "index.html"), html)
        { "html" => File.join(@directory, "index.html"), "loss_png" => File.join(@directory, "loss.png"),
          "loss_svg" => File.join(@directory, "loss.svg"), "console" => File.join(@directory, "loss.txt") }
      end

      private

      def trace(stage, filename, key, optional: false)
        path = File.join(@root, stage, filename)
        return [] unless File.file?(path)
        # Retries/resumes can repeat a step. Keep its latest observation.
        rows = {}
        File.foreach(path) do |line|
          next if line.strip.empty?
          row = JSON.parse(line)
          step, value = row.fetch("step"), optional ? row[key] : row.fetch(key)
          next if optional && value.nil?
          raise ArgumentError, "Invalid plot observation in #{path}" unless step.is_a?(Integer) && step > 0 && value.is_a?(Numeric) && value.finite?
          rows[step] = value
        end
        rows.sort
      end

      def read_json(relative)
        path = File.join(@root, relative)
        JSON.parse(File.read(path)) if File.file?(path)
      end

      def write_series
        @series.each do |stage, series|
          series.each do |name, values|
            File.write(File.join(@directory, "#{stage}-#{name}.dat"), values.map { |step, loss| "#{step}\t#{loss}" }.join("\n") + "\n")
          end
        end
        @memory.each do |stage, series|
          series.each do |name, values|
            File.write(File.join(@directory, "#{stage}-memory-#{name}.dat"), values.map { |step, value| "#{step}\t#{value}" }.join("\n") + "\n")
          end
        end
        return unless @calibration && @test
        data = %w[nll brier ece].each_with_index.map do |key, index|
          "#{index}\t#{@calibration.fetch('before').fetch(key)}\t#{@calibration.fetch('after').fetch(key)}"
        end
        File.write(File.join(@directory, "calibration.dat"), data.join("\n") + "\n")
        reliability = @test.fetch("reliability").map { |bin| "#{bin.fetch('confidence')}\t#{bin.fetch('accuracy')}\t#{bin.fetch('count')}" }
        File.write(File.join(@directory, "reliability.dat"), reliability.join("\n") + "\n")
      end

      def write_console
        charts = @series.filter_map do |stage, series|
          next if series["train"].empty?
          xs, ys = series["train"].transpose
          values = series.values.flatten(1)
          minimum, maximum = values.map(&:last).minmax
          padding = [maximum - minimum, 0.1].max * 0.08
          method = xs.length == 1 ? :scatterplot : :lineplot
          plot = UnicodePlot.public_send(method, xs, ys, title: "#{stage.upcase} loss", xlabel: "Optimizer step", ylabel: "Loss",
            name: "train", width: 65, height: 12, color: :blue,
            xlim: [0, values.map(&:first).max + 1], ylim: [[minimum - padding, 0].max, maximum + padding])
          plot.annotate_row!(:l, 0, format("%.4f", plot.origin_y + plot.plot_height))
          plot.annotate_row!(:l, 11, format("%.4f", plot.origin_y))
          unless series["validation"].empty?
            vx, vy = series["validation"].transpose
            method = vx.length == 1 ? :scatterplot! : :lineplot!
            UnicodePlot.public_send(method, plot, vx, vy, name: "validation", color: :red)
          end
          plot.to_s.gsub(/\e\[[0-9;]*m/, "")
        end
        text = charts.join("\n\n")
        File.write(File.join(@directory, "loss.txt"), text + "\n")
        @out.puts(text)
      end

      def render(name, body)
        script = <<~GNUPLOT
          set encoding utf8
          set border 3 lc rgb '#64748b'
          set tics nomirror
          set grid back lc rgb '#e2e8f0'
          set key outside top center horizontal opaque box lc rgb '#e2e8f0'
          set terminal pngcairo size 1280,520 enhanced font 'Sans,11'
          set output '#{name}.png'
          #{body}
          unset output
          set terminal svg size 1280,520 enhanced font 'sans,11'
          set output '#{name}.svg'
          #{body}
          unset output
        GNUPLOT
        path = File.join(@directory, "#{name}.gnuplot")
        File.write(path, script)
        _, error, status = Open3.capture3("gnuplot", "#{name}.gnuplot", chdir: @directory)
        raise ArgumentError, "Chart generation failed: #{error}" unless status.success?
      end

      def loss_script
        panels = @series.filter_map do |stage, series|
          next if series["train"].empty?
          lines = ["'#{stage}-train.dat' using 1:2 with linespoints lw 1.5 pt 7 ps 0.2 lc rgb '#2563eb' title 'Train (sampled batch)'"]
          unless series["validation"].empty?
            lines << "'#{stage}-validation.dat' using 1:2 with linespoints lw 2 pt 7 ps 0.6 lc rgb '#ea580c' title 'Validation (held out)'"
          end
          selected = @summary.fetch("selected_steps", {})[stage]
          selected = nil unless selected.is_a?(Integer) && selected > 0
          <<~GNUPLOT
            set title '#{stage == 'mlm' ? 'MLM pretraining' : 'Candidate choice training'}#{selected ? " (selected step #{selected})" : ''}'
            set xlabel 'Optimizer step'
            set ylabel '#{stage == 'mlm' ? 'Masked token cross-entropy' : 'Candidate cross-entropy'}'
            set xrange [0:#{series.values.flatten(1).map(&:first).max + 1}]
            set yrange [0:*]
            #{selected ? "set arrow 1 from #{selected}, graph 0 to #{selected}, graph 1 nohead dt 3 lc rgb '#16a34a'" : ''}
            plot #{lines.join(', ')}
            unset arrow
          GNUPLOT
        end
        "set multiplot layout 1,#{panels.length} title 'EasyAI Decision - observed loss (unsmoothed)'\n#{panels.join("\n")}\nunset multiplot\n"
      end

      def evaluation_script
        <<~GNUPLOT
          set multiplot layout 1,2 title 'EasyAI Decision - calibration fit and held-out test'
          set title 'Calibration split: lower is better'
          unset xlabel
          set ylabel 'Metric value (different scales)'
          set xrange [-0.6:2.6]
          set yrange [0:*]
          set xtics ('NLL' 0, 'Brier' 1, 'ECE' 2)
          set boxwidth 0.32
          set style fill solid 0.85 border -1
          plot 'calibration.dat' using ($1-0.17):2 with boxes lc rgb '#94a3b8' title 'Before temperature', \
               'calibration.dat' using ($1+0.17):3 with boxes lc rgb '#2563eb' title 'After temperature'
          set title 'Test reliability: closer to diagonal is better'
          set xlabel 'Mean confidence in bin'
          set ylabel 'Accuracy in bin'
          set xrange [0:1]
          set yrange [0:1]
          set xtics autofreq
          plot x with lines dt 2 lc rgb '#94a3b8' title 'Ideal', \
               'reliability.dat' using 1:2 with points pt 7 ps 1.3 lc rgb '#ea580c' title 'Observed bins'
          unset multiplot
        GNUPLOT
      end

      def memory_script
        panels = @memory.map do |stage, series|
          lines = series.filter_map do |point, values|
            next if values.empty?
            "'#{stage}-memory-#{point}.dat' using 1:2 with linespoints lw 2 pt #{point == 'before' ? 6 : 7} title '#{point} validation'"
          end
          <<~GNUPLOT
            set title '#{stage.upcase}: GPU process footprint'
            set xlabel 'Optimizer step'
            set ylabel 'MiB (includes driver / allocator caches)'
            set xrange [0:#{series.values.flatten(1).map(&:first).max + 1}]
            set yrange [0:*]
            plot #{lines.join(', ')}
          GNUPLOT
        end
        "set multiplot layout 1,#{panels.length} title 'Validation memory: should plateau after warmup'\n#{panels.join("\n")}\nunset multiplot\n"
      end

      def html
        summary = %w[train validation calibration test].filter_map do |split|
          info = @summary.fetch("data", {})[split]
          "#{split}: #{info['rows']} rows / #{info['groups']} groups" if info
        end.join("; ")
        metrics = if @test
          rows = [["all", @test]] + @test.fetch("by_language", {}).to_a + @test.fetch("by_source", {}).map { |name, row| ["task: #{name}", row] }
          rows.map do |language, row|
            cells = [CGI.escapeHTML(language), row.fetch("count"), *%w[accuracy nll brier ece].map { |key| format("%.5f", row.fetch(key)) }]
            "<tr>#{cells.map { |value| "<td>#{value}</td>" }.join}</tr>"
          end.join("\n")
                  else
          '<tr><td colspan="6">Test evaluation has not completed.</td></tr>'
                  end
        <<~HTML
          <!doctype html>
          <html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
          <title>EasyAI Decision training report</title>
          <style>
            body{font:16px/1.6 system-ui,sans-serif;color:#0f172a;background:#f8fafc;max-width:1280px;margin:32px auto;padding:0 24px}
            img{width:100%;background:white;border:1px solid #e2e8f0;border-radius:8px}
            table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px;border-bottom:1px solid #e2e8f0;text-align:right}
            th:first-child,td:first-child{text-align:left}code{overflow-wrap:anywhere}a{color:#2563eb}
          </style>
          <h1>EasyAI Decision 训练报告</h1>
          <p>Run: <code>#{CGI.escapeHTML(@root)}</code></p>
          <p>#{CGI.escapeHTML(summary)}</p>
          <p>训练曲线记录采样 batch 的 loss；验证曲线使用独立 validation split。未做平滑。
          翻译样本可能共享同一语义分组，行数不等于独立样本数。低训练 loss 不代表通用判断能力。</p>
          <p>#{selection_description}</p>
          <h2>Loss / 优化器更新步数</h2>
          <img src="loss.svg" alt="MLM and candidate training and validation loss">
          <p><a href="loss.png">PNG</a> · <a href="loss.svg">SVG</a> · <a href="loss.txt">终端曲线</a></p>
          #{@memory.empty? ? '' : '<h2>验证前后显存</h2><p>进程占用含驱动与分配器缓存；预热后应稳定，不要求降为零。这是验证边界测量，不是 batch 内峰值。</p><img src="memory.svg" alt="GPU process memory before and after validation"><p><a href="memory.png">PNG</a> · <a href="memory.svg">SVG</a></p>'}
          #{diagnostic_html}
          #{semantic_diagnostic_html}
          <h2>Held-out test</h2>
          <table><thead><tr><th>Language</th><th>Rows</th><th>Accuracy</th><th>NLL</th><th>Brier</th><th>ECE</th></tr></thead><tbody>#{metrics}</tbody></table>
          #{@calibration && @test ? '<h2>温度校准与可靠性</h2><img src="evaluation.svg" alt="Calibration metrics and held-out test reliability"><p><a href="evaluation.png">PNG</a> · <a href="evaluation.svg">SVG</a></p>' : ''}
          <p>温度以 calibration NLL 为拟合目标；ECE 不保证改善。Test 数据没有参与拟合温度。</p>
          <p><a href="../summary.json">运行信息</a> · <a href="../config.yml">配置</a> · <a href="../pipeline.log">进度日志</a> · <a href="../train.log">逐步训练日志</a></p>
          </html>
        HTML
      end

      def diagnostic_html
        return "" unless @diagnostic
        rows = @diagnostic.fetch("conditions").map do |condition, metrics|
          values = [CGI.escapeHTML(condition), *%w[accuracy nll prediction_agreement_with_original mean_max_probability_change].map { |key| format("%.5f", metrics.fetch(key)) }]
          "<tr>#{values.map { |value| "<td>#{value}</td>" }.join}</tr>"
        end.join
        <<~HTML
          <h2>状态依赖诊断（validation）</h2>
          <p>#{CGI.escapeHTML(@diagnostic.fetch('scope'))}。保留候选，替换状态。
          如果预测几乎不变，或原始状态不优于标签频率基线，模型可能只学到了答案偏好。
          预测发生变化也不等于语义正确，需要结合准确率与任务测试判断。</p>
          <table><thead><tr><th>Condition</th><th>Accuracy</th><th>NLL</th><th>与原预测一致率</th><th>平均最大概率变化</th></tr></thead><tbody>#{rows}</tbody></table>
          <p><a href="../stage-results/diagnostic.json">完整诊断</a></p>
        HTML
      end

      def selection_description
        return "旧运行未记录最佳权重选择信息，请查阅该实验的运行记录。" unless @summary["selected_steps"]
        steps = @summary["selected_steps"].filter_map { |stage, step| "#{stage}=#{step}" if step }.join(", ")
        "验证集选中的权重步数：#{CGI.escapeHTML(steps)}。最后一步保留优化器以续训；校准和测试使用验证集 loss 最低的权重。"
      end

      def semantic_diagnostic_html
        return "" unless @semantic_diagnostic
        rows = @semantic_diagnostic.fetch("by_source_language").flat_map do |source, result|
          result.fetch("conditions").map do |condition, metrics|
            cells = [CGI.escapeHTML(source), CGI.escapeHTML(condition), *%w[accuracy nll prediction_agreement].map { |key| format("%.5f", metrics.fetch(key)) }]
            "<tr>#{cells.map { |value| "<td>#{value}</td>" }.join}</tr>"
          end
        end.join
        <<~HTML
          <h2>各语义任务：状态与问题消融</h2>
          <p>固定候选，在同任务、同语言内打乱状态或问题。来自 validation，不是 test。预测变化本身不代表正确理解。</p>
          <table><thead><tr><th>Source / language</th><th>Condition</th><th>Accuracy</th><th>NLL</th><th>预测一致率</th></tr></thead><tbody>#{rows}</tbody></table>
        HTML
      end
    end
  end
end
