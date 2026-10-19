module DecisionRelational
  RECIPES = {
    "separate-rotary" => { "reference" => "separate", "model" => { "position_encoding" => "rotary" }, "hypothesis" => "Test whether explicit state/query interaction plus RoPE retains actor relations better on the broader controlled core." },
    "joint-rotary-candidate" => { "reference" => "joint-rotary", "model" => { "pooling" => "candidate" }, "hypothesis" => "After broader actor binding fails under mean pooling, test candidate-only readout on the repeatably fitting RoPE representation." },
      "joint-candidate" => { "model" => { "pooling" => "candidate" }, "hypothesis" => "All-token averaging dilutes the option-conditioned relation; pool only actual candidate tokens." },
      "joint-rotary" => { "model" => { "position_encoding" => "rotary" }, "hypothesis" => "Relative positional attention may preserve actor/negation binding better than transferred additive positions." },
      "joint-position-scale" => { "model" => { "position_scale" => 0.2 }, "hypothesis" => "The parent's 0.02 additive position signal may be too weak for order-sensitive supervision; test one fixed tenfold scale." },
      "joint-scratch" => { "model" => {}, "initialization" => "scratch", "hypothesis" => "The own-parent representation may resist relational fitting; test identical joint architecture with fresh deterministic tensors and the same tokenizer." }
    }.freeze
  module_function

  def freeze_recipe(root, name)
    protocol = verify(root)
    authorize_pilot_failure!(root, "focused") if %w[joint-rotary-candidate separate-rotary].include?(name)
    recipe = RECIPES.fetch(name) { raise ArgumentError, "Unknown bounded follow-up" }
    ARMS.each do |arm|
      path = File.join(root, "fit-#{arm}-1337/result.json")
      raise "Primary comparison incomplete" unless File.exist?(path)
      raise "Primary fit passed; follow its confirmation before rescue fits" if JSON.parse(File.read(path)).fetch("passed")
    end
    directory = File.join(root, "recipes")
    FileUtils.mkdir_p(directory)
    path = File.join(directory, "#{name}.json")
    raise "Recipe already frozen" if File.exist?(path)
    reference = recipe.fetch("reference", "joint")
    base, = recipe_config(root, protocol, reference)
    config = EasyAI::Decision::Config.new(base).with(model: recipe.fetch("model"))
    value = { "name" => name, "reference" => reference, "hypothesis" => recipe.fetch("hypothesis"),
      "initialization" => recipe.fetch("initialization", "parent"), "config" => config.to_h,
      "protocol_sha256" => sha(File.join(root, "protocol.json")), "created_at" => Time.now.utc.iso8601,
      "source_sha256" => Dir.glob("{lib/easy_ai,benchmarks/decision}/**/*.rb").sort.to_h { |file| [file, sha(file)] },
      "scope" => "One finite post-v0.5 fitting intervention. No acceptance observations used; maximum 1000 updates, unchanged fit gates." }
    value.fetch("source_sha256").each do |file, digest|
      raise "Source changed during recipe freeze" unless sha(file) == digest
      destination = File.join(root, "source-#{name}-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
    end
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end

  def recipe_config(root, protocol, arm)
    return [protocol.fetch("configs").fetch(arm), "parent"] if ARMS.include?(arm)
    raise ArgumentError, "Unknown bounded follow-up" unless RECIPES.key?(arm)
    path = File.join(root, "recipes", "#{arm}.json")
    digest = JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    raise "Frozen recipe changed" unless sha(path) == digest
    recipe = JSON.parse(File.read(path))
    raise "Recipe protocol changed" unless recipe.fetch("protocol_sha256") == sha(File.join(root, "protocol.json"))
    [recipe.fetch("config"), recipe.fetch("initialization")]
  end
end
