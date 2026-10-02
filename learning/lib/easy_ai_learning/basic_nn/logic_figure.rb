require "fileutils"
require "open3"

module EasyAILearning
  module BasicNN
    # Gnuplot is optional: use it only for a static image embedded in documentation.
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
        File.join(directory, "functions.png")
      end

      private

      def write_data(directory)
        coordinates = (0..120).map { |index| index / 100.0 }
        inputs = coordinates.flat_map { |x2| coordinates.map { |x1| [x1, x2] } }
        scores = @model.batch_scores(inputs)
        rows = inputs.zip(scores).each_slice(coordinates.size).map do |slice|
          slice.map { |input, output| (input + output).join("\t") }.join("\n")
        end
        File.write(File.join(directory, "functions.dat"), rows.join("\n\n") + "\n")
        truth = LogicGates::INPUTS.each_with_index.map { |input, row| (input + LogicGates.targets[row]).join("\t") }
        File.write(File.join(directory, "truth.dat"), truth.join("\n") + "\n")
        xs = (-100..220).map { |index| index / 100.0 }
        slice = @model.batch_scores(xs.map { |x| [x, 0] })
        curve = xs.zip(slice).map { |x, scores| "#{x}\t#{scores[3]}" }
        File.write(File.join(directory, "xor-curve.dat"), curve.join("\n") + "\n")
        File.write(File.join(directory, "loss.dat"), @history.map { |step, loss| "#{step}\t#{[loss, 1e-16].max}" }.join("\n") + "\n")
      end

      def script
        <<~GNUPLOT
          set format x '%.17g'
          set format y '%.17g'
          set format z '%.17g'
          set contour base
          set cntrparam levels discrete 0.5
          unset surface
          #{LogicGates::NAMES.each_index.map { |gate| "set table 'boundary-#{gate}.dat'\nsplot 'functions.dat' using 1:2:#{gate + 3}\nunset table" }.join("\n")}
          unset contour
          set surface
          set format x '%g'
          set format y '%g'
          set terminal pngcairo size 1440,900 enhanced font 'Sans,11'
          set output 'functions.png'
          set encoding utf8
          set border 3
          set tics nomirror
          set grid back lc rgb '#e2e8f0'
          set key top center outside horizontal font ',9'
          set multiplot layout 2,3 rowsfirst title 'Torch.rb learning: 2 -> 2 ReLU -> 4 linear, 18 trained parameters | Seed #{@metadata.fetch("seed")}, #{@metadata.fetch("steps")} updates, device #{@metadata.fetch("device")}; curves use saved learned weights'
          set xrange [0:1.2]
          set yrange [0:1.2]
          set xtics 0.2
          set ytics 0.2
          set palette defined (0 '#dbeafe', 1 '#ffedd5')
          set cbrange [0:1]
          unset colorbox
          set xlabel 'x1'
          set ylabel 'x2'
          #{LogicGates::NAMES.each_index.map { |gate| gate_script(gate) }.join("\n")}
          set title 'Hidden activation: ReLU(x) = max(0,x)'
          set xrange [-2:2]
          set yrange [-0.1:2.1]
          set xtics autofreq
          set ytics autofreq
          set xlabel 'Preactivation'
          set ylabel 'Activation'
          plot (x > 0 ? x : 0) lw 2 lc rgb '#16a34a' notitle
          set xrange [0:1.2]
          set yrange [0:1.2]
          set xtics 0.2
          set ytics 0.2
          set xlabel 'x1'
          set ylabel 'x2'
          #{focused_xor_script}
          unset multiplot
          unset output
        GNUPLOT
      end

      def focused_xor_script
        <<~GNUPLOT
          set title 'Saved trained model: XOR regions and boundary'
          set palette defined (0 '#dbeafe', 1 '#dcfce7')
          set label 1 'XOR = 1' at 0.58,0.60 center front textcolor rgb '#166534'
          set label 2 'same positive region' at 0.58,0.52 center front font ',9' textcolor rgb '#166534'
          plot 'functions.dat' using 1:2:($6 >= 0.5 ? 1 : 0) with image notitle, \
            'boundary-3.dat' using 1:2 with lines lw 2.5 lc rgb '#111827' title 'model boundary', \
            'truth.dat' using 1:($6 == 0 ? $2 : 1/0) with points pt 7 ps 1.8 lc rgb '#2563eb' title 'XOR=0', \
            'truth.dat' using 1:($6 == 1 ? $2 : 1/0) with points pt 7 ps 1.8 lc rgb '#16a34a' title 'XOR=1', \
            'truth.dat' using 1:2:(sprintf('(%g,%g):%g', $1, $2, $6)) with labels left offset 1,1 textcolor rgb '#111827' notitle
          unset label 1
          unset label 2
        GNUPLOT
      end

      def gate_script(gate)
        column = gate + 3
        background = "'functions.dat' using 1:2:($#{column} >= 0.5 ? 1 : 0) with image notitle, "
        title = "#{LogicGates::NAMES[gate].upcase}: learned decision boundary, score = 0.5"
        <<~GNUPLOT
          set title '#{title}'
          plot #{background}'boundary-#{gate}.dat' using 1:2 with lines lw 2 lc rgb '#111827' title 'boundary', \
            'truth.dat' using 1:($#{column} == 0 ? $2 : 1/0) with points pt 7 ps 1.5 lc rgb '#2563eb' title 'truth 0', \
            'truth.dat' using 1:($#{column} == 1 ? $2 : 1/0) with points pt 7 ps 1.5 lc rgb '#ea580c' title 'truth 1', \
            'truth.dat' using 1:2:(sprintf('(%g,%g):%g', $1, $2, $#{column})) with labels offset 1,1 textcolor rgb '#111827' notitle
        GNUPLOT
      end
    end
  end
end
