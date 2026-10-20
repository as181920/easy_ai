module DecisionScaling
  module_function

  def scaling_plot(root)
    scaling_report(root)
    directory = File.join(root, "plot")
    FileUtils.mkdir_p(directory)
    series = Dir[File.join(root, "arms/*/development.json")].sort.map.with_index do |path, index|
      history = JSON.parse(File.read(path))
      rows = history.map do |cell|
        [cell.fetch("step"), cell.dig("training_probe", "worst_language_balanced"), cell.dig("canonical", "worst_language_balanced"),
          minimum_groups(cell.fetch("training_probe")), minimum_groups(cell.fetch("canonical"))].join(" ")
      end
      target = File.join(directory, "series-#{index}.dat")
      File.write(target, rows.join("\n") + "\n")
      [File.basename(File.dirname(path)), target, index + 1]
    end
    raise "No scaling development measurements" if series.empty?
    quote = ->(value) { '"' + value.to_s.gsub('\\', '\\\\').gsub('"', '\\"') + '"' }
    curves = ->(column) { series.map { |name, path, color| "#{quote.call(path)} using 1:#{column} with linespoints lc #{color} title #{quote.call(name)}" }.join(", \\\n") }
    output = File.expand_path("docs/images/decision-v06-scaling.png")
    script = <<~GP
      set terminal pngcairo size 1800,1200 font "Sans,11"
      set output #{quote.call(output)}
      set multiplot layout 2,2 margins 0.07,0.98,0.08,0.92 spacing 0.08,0.12 title "Decision scaling: initial matched-visit screen and paired 512-world continuations (development)"
      set grid
      set xlabel "Optimizer updates"
      set yrange [0:1.05]
      set key bottom right
      set arrow 1 from 1500, graph 0 to 1500, graph 1 nohead dt 2 lw 1.2 lc rgb "#555555"
      set ylabel "Balanced accuracy"
      set title "Worst-language training-probe accuracy (16 worlds/language)"
      plot #{curves.call(2)}, 0.95 with lines dt 2 lc rgb "gray" title "Mastery accuracy floor"
      set title "Worst-language canonical binary development accuracy"
      plot #{curves.call(3)}, 0.75 with lines dt 2 lc rgb "gray" title "Development accuracy floor"
      set ylabel "All-correct fraction"
      set key top right
      set title "Worst actor/event training-probe group correctness"
      plot #{curves.call(4)}, 0.9 with lines dt 2 lc rgb "gray" title "Historical retention 90%", 0.85 with lines dt 3 lc rgb "#555555" title "Development retention 85%"
      set title "Worst actor/event canonical development group correctness"
      plot #{curves.call(5)}, 0.5 with lines dt 2 lc rgb "gray" title "Development group floor"
      unset multiplot
    GP
    File.write(File.join(directory, "figure.gp"), script)
    IO.popen(["gnuplot"], "w") { |process| process.write(script) }
    raise "gnuplot failed" unless $?.success?
    puts "Wrote #{output}"
  end
end
