require "fileutils"
require "open3"

module EasyAILearning
  module BasicNN
    # Fixed logic references and saved XOR model outputs are plotted in separate groups.
    class LogicFigure
      def initialize(model:, history:, metadata:)
        @model, @history, @metadata = model, history, metadata
      end

      def write(directory)
        FileUtils.mkdir_p(directory)
        write_data(directory)
        File.write(File.join(directory, "functions.gnuplot"), script)
        _, error, status = Open3.capture3("gnuplot", "functions.gnuplot", chdir: directory)
        raise "Gnuplot failed: #{error}" unless status.success?
        %w[logic-functions model-functions score-heatmap score-surface].map { |name| File.join(directory, "#{name}.png") }
      end

      private

      def write_data(directory)
        coordinates = (0..120).map { |index| index / 100.0 }
        inputs = coordinates.flat_map { |x2| coordinates.map { |x1| [x1, x2] } }
        write_references(directory, inputs, coordinates.size)
        scores = @model.batch_scores(inputs)
        rows = inputs.zip(scores).each_slice(coordinates.size).map do |slice|
          slice.map { |input, output| (input + output).join("\t") }.join("\n")
        end
        File.write(File.join(directory, "functions.dat"), rows.join("\n\n") + "\n")
        probes = LogicGates::INPUTS + [[0.5, 0.5], [0.123, 0.123]]
        probe_rows = probes.zip(@model.batch_scores(probes)).map { |input, output| (input + output).join("\t") }
        File.write(File.join(directory, "score-points.dat"), probe_rows.join("\n") + "\n")
        truth = LogicGates::INPUTS.each_with_index.map { |input, row| (input + LogicGates.targets[row]).join("\t") }
        File.write(File.join(directory, "truth.dat"), truth.join("\n") + "\n")
        File.write(File.join(directory, "loss.dat"), @history.map { |step, loss| "#{step}\t#{[loss, 1e-16].max}" }.join("\n") + "\n")
      end

      def write_references(directory, inputs, width)
        exact = ScalarLogicNetwork.exact
        rows = inputs.each_slice(width).map do |slice|
          slice.map do |input|
            scores = LogicGates::PERCEPTRONS.values.map { |config| input.zip(config[:weights]).sum { |x, w| x * w } + config[:bias] }
            (input + scores + exact.forward(input)).join("\t")
          end.join("\n")
        end
        File.write(File.join(directory, "references.dat"), rows.join("\n\n") + "\n")
        truth = LogicGates::INPUTS.map { |input| (input + LogicGates::NAMES.map { |name| LogicGates.call(name, *input) }).join("\t") }
        File.write(File.join(directory, "reference-truth.dat"), truth.join("\n") + "\n")
      end

      def reference_contours
        LogicGates::NAMES.each_index.map do |gate|
          "set cntrparam levels discrete #{gate == 3 ? 0.5 : 0}\nset table 'reference-boundary-#{gate}.dat'\nsplot 'references.dat' using 1:2:#{gate + 3}\nunset table"
        end.join("\n")
      end

      def reference_panels
        LogicGates::NAMES.each_with_index.map do |name, gate|
          column, threshold = gate + 3, gate == 3 ? 0.5 : 0
          <<~GNUPLOT
            set title '#{name.upcase}: fixed mathematical function'
            plot 'references.dat' using 1:2:($#{column} >= #{threshold} ? 1 : 0) with image notitle, \
              'reference-boundary-#{gate}.dat' using 1:2 with lines lw 2 lc rgb '#111827' title 'reference boundary', \
              'reference-truth.dat' using 1:($#{column} == 0 ? $2 : 1/0) with points pt 7 ps 1.5 lc rgb '#2563eb' title 'truth 0', \
              'reference-truth.dat' using 1:($#{column} == 1 ? $2 : 1/0) with points pt 7 ps 1.5 lc rgb '#16a34a' title 'truth 1', \
              'reference-truth.dat' using 1:2:(sprintf('(%g,%g):%g', $1, $2, $#{column})) with labels left offset 1,1 notitle
          GNUPLOT
        end.join("\n")
      end

      def score_style
        <<~GNUPLOT
          unset logscale y
          set xrange [0:1.2]
          set yrange [0:1.2]
          set xtics 0.2
          set ytics 0.2
          set xlabel 'x1'
          set ylabel 'x2'
          set cbrange [score_min:score_max]
          set palette defined (0 '#440154', 0.25 '#3b528b', 0.5 '#21918c', 0.75 '#5ec962', 1 '#fde725')
          set colorbox
          set cblabel 'Raw score'
          set format cb '%g'
          set key top center outside horizontal font ',9'
        GNUPLOT
      end

      def score_heatmap
        <<~GNUPLOT
          #{score_style}
          set title 'Raw XOR score: heatmap and contours'
          plot 'functions.dat' using 1:2:3 with image notitle, \
            'score-contours.dat' using 1:2 with lines lw 1 lc rgb '#ffffff' title '0 / .25 / .5 / .75 / 1', \
            'boundary.dat' using 1:2 with lines lw 2 lc rgb '#111827' title 'score = 0.5', \
            'score-points.dat' using 1:2 with points pt 7 ps 1 lc rgb '#ef4444' notitle, \
            'score-points.dat' using 1:2:(sprintf('(%g,%g)', $1, $2)) with labels offset 1,1 textcolor rgb '#111827' notitle
        GNUPLOT
      end

      def score_surface
        <<~GNUPLOT
          #{score_style}
          set title 'Raw XOR score: 3D surface'
          set zlabel 'Raw score' rotate parallel
          set zrange [*:*]
          set ztics autofreq
          set view 60,35,1,1
          set xyplane relative 0
          set pm3d depthorder
          unset key
          splot 'functions.dat' using 1:2:3 with pm3d notitle, \
            'score-points.dat' using 1:2:3 with points pt 7 ps 1.3 lc rgb '#ef4444' notitle
          unset pm3d
          unset zlabel
        GNUPLOT
      end

      def script
        <<~GNUPLOT
          set format x '%.17g'
          set format y '%.17g'
          set format z '%.17g'
          set contour base
          set cntrparam levels discrete 0.5
          unset surface
          #{reference_contours}
          set cntrparam levels discrete 0.5
          set table 'boundary.dat'
          splot 'functions.dat' using 1:2:3
          unset table
          set cntrparam levels discrete 0,0.25,0.5,0.75,1
          set table 'score-contours.dat'
          splot 'functions.dat' using 1:2:3
          unset table
          unset contour
          set surface
          set format x '%g'
          set format y '%g'
          set format z '%g'
          set terminal pngcairo size 1000,900 enhanced font 'Sans,11'
          set encoding utf8
          set border 3
          set tics nomirror
          set grid back lc rgb '#e2e8f0'
          set key top center outside horizontal font ',9'
          set output 'logic-functions.png'
          set multiplot layout 2,2 rowsfirst title 'Fixed logic functions: AND / OR / NAND / XOR (not trained model outputs)'
          set xrange [0:1.2]
          set yrange [0:1.2]
          set xtics 0.2
          set ytics 0.2
          set palette defined (0 '#dbeafe', 1 '#dcfce7')
          set cbrange [0:1]
          unset colorbox
          set xlabel 'x1'
          set ylabel 'x2'
          #{reference_panels}
          unset multiplot
          unset output
          set xrange [*:*]
          set yrange [*:*]
          stats 'functions.dat' using 3 nooutput
          score_min = STATS_min
          score_max = STATS_max
          set terminal pngcairo size 1440,1000 enhanced font 'Sans,11'
          set output 'model-functions.png'
          set multiplot title 'Trained XOR model: 2 -> 2 ReLU -> 1, 9 parameters | Seed #{@metadata.fetch("seed")}, #{@metadata.fetch("steps")} updates, #{@metadata.fetch("device")}'
          set size 0.5,0.46
          set origin 0,0.5
          set title 'Hidden activation: ReLU(x) = max(0,x)'
          set xrange [-2:2]
          set yrange [-0.1:2.1]
          set xtics autofreq
          set ytics autofreq
          set xlabel 'Preactivation'
          set ylabel 'Activation'
          plot (x > 0 ? x : 0) lw 2 lc rgb '#16a34a' notitle
          set origin 0.5,0.5
          set title 'XOR training MSE (log scale): #{@metadata.fetch("steps")} #{@metadata.fetch("device")} updates'
          set xrange [0:#{@metadata.fetch("steps")}]
          set yrange [*:*]
          set logscale y
          set format y '%g'
          set xlabel 'Optimizer update'
          set ylabel 'MSE'
          plot 'loss.dat' using 1:2 with lines lw 2 lc rgb '#2563eb' notitle
          set size 0.333333,0.47
          set origin 0,0
          #{score_heatmap}
          set origin 0.333333,0
          #{score_surface}
          set origin 0.666666,0
          unset colorbox
          set palette defined (0 '#dbeafe', 1 '#dcfce7')
          set cbrange [0:1]
          set key top center outside horizontal font ',9'
          unset logscale y
          set title 'Saved XOR model: score = 0.5 boundary'
          set xrange [0:1.2]
          set yrange [0:1.2]
          set xtics 0.2
          set ytics 0.2
          set xlabel 'x1'
          set ylabel 'x2'
          plot 'functions.dat' using 1:2:($3 >= 0.5 ? 1 : 0) with image notitle, \
            'boundary.dat' using 1:2 with lines lw 2.5 lc rgb '#111827' title 'model boundary', \
            'truth.dat' using 1:($3 == 0 ? $2 : 1/0) with points pt 7 ps 1.8 lc rgb '#2563eb' title 'XOR=0', \
            'truth.dat' using 1:($3 == 1 ? $2 : 1/0) with points pt 7 ps 1.8 lc rgb '#16a34a' title 'XOR=1', \
            'truth.dat' using 1:2:(sprintf('(%g,%g):%g', $1, $2, $3)) with labels left offset 1,1 textcolor rgb '#111827' notitle
          unset multiplot
          unset output
          set size 1,1
          set origin 0,0
          set terminal pngcairo size 820,720 enhanced font 'Sans,11'
          set output 'score-heatmap.png'
          #{score_heatmap}
          unset output
          set terminal pngcairo size 900,720 enhanced font 'Sans,11'
          set output 'score-surface.png'
          #{score_surface}
          unset output
        GNUPLOT
      end
    end
  end
end
