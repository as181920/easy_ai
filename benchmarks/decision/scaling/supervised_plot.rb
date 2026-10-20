module DecisionScaling
  module_function

  def supervised_plot(root)
    protocol = verify(root)
    directory = File.join(root, "plot-supervised")
    FileUtils.mkdir_p(directory)
    histories = Dir[File.join(root, "supervised/*/development.json")].sort
    raise "No supervised development measurements" if histories.empty?
    maximum_step = 0
    series = histories.map.with_index do |path, index|
      history = JSON.parse(File.read(path))
      points = history.map do |cell|
        maximum_step = [maximum_step, cell.fetch("step")].max
        binary = cell.fetch("profiles").fetch("binary")
        joint = cell.fetch("profiles")["joint"]
        summarize = lambda do |profile|
          next ["NaN", "NaN", "NaN"] unless profile
          required = protocol.fetch("required_development").map { |name| profile.fetch("controlled").fetch(name) }
          [profile.dig("controlled", "canonical", "worst_language_balanced"), required.map { |metrics| minimum_groups(metrics) }.min,
            profile.fetch("natural").fetch("by_language").values.map { |metrics| metrics.fetch("accuracy") }.min]
        end
        [cell.fetch("step"), cell.dig("training_probe", "worst_language_balanced"), *summarize.call(binary), *summarize.call(joint)].join(" ")
      end
      target = File.join(directory, "series-#{index}.dat")
      File.write(target, points.join("\n") + "\n")
      [File.basename(File.dirname(path)), target, index + 1, history.any? { |cell| cell.fetch("profiles").key?("joint") }]
    end
    quote = ->(value) { '"' + value.to_s.gsub('\\', '\\\\').gsub('"', '\\"') + '"' }
    curves = lambda do |column, joint_column = nil|
      series.flat_map do |name, path, color, joint_measured|
        values = ["#{quote.call(path)} using 1:#{column} with linespoints lc #{color} title #{quote.call(name + '/binary')}"]
        values << "#{quote.call(path)} using 1:#{joint_column} with linespoints dt 2 lc #{color} title #{quote.call(name + '/joint')}" if joint_column && joint_measured
        values
      end.join(", \\\n")
    end
    output = File.expand_path("docs/images/decision-v06-supervised.png")
    script = <<~GP
      set terminal pngcairo size 1800,1200 font "Sans,11"
      set output #{quote.call(output)}
      set multiplot layout 2,2 margins 0.07,0.98,0.08,0.92 spacing 0.08,0.12 title "Conditional scale supervision: observed development, not independent acceptance"
      set grid
      set xlabel "Additional supervised optimizer updates"
      set xrange [0:#{maximum_step}]
      set yrange [0:1.05]
      set key bottom right
      set arrow 1 from 500, graph 0 to 500, graph 1 nohead dt 2 lw 1.2 lc rgb "#555555"
      set title "Worst-language known training-probe accuracy"
      set ylabel "Balanced accuracy"
      plot #{curves.call(2)}, 0.95 with lines dt 2 lc rgb "gray" title "Known mastery accuracy floor"
      set title "Worst-language canonical controlled development"
      plot #{curves.call(3, 6)}
      set title "Worst actor/event group across all required controlled views"
      set key top right
      set ylabel "All-correct fraction"
      plot #{curves.call(4, 7)}, 0.5 with lines dt 2 lc rgb "gray" title "Development group floor"
      set title "Fresh natural development (profile-specific reviewed panels)"
      set ylabel "Accuracy"
      plot #{curves.call(5, 8)}, 0.65 with lines dt 2 lc rgb "gray" title "Natural accuracy floor"
      unset multiplot
    GP
    File.write(File.join(directory, "figure.gp"), script)
    IO.popen(["gnuplot"], "w") { |process| process.write(script) }
    raise "gnuplot failed" unless $?.success?
    puts "Wrote #{output}"
  end
end
