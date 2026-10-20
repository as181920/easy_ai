require_relative '../event_generalization'

if $PROGRAM_NAME == __FILE__
  a=DecisionEventGeneralization
  root=File.expand_path(ARGV.fetch(0, 'runs/decision/event-generalization-v08'))
  destination = File.expand_path(ARGV.fetch(1, root))
  rows=a.raw(root, 'train')
  examples=rows.map { |row| EasyAI::Decision::Data::Example.new(row) }
  groups=rows.each_index.group_by { |i| r=rows[i]; [r.fetch('language'), r.fetch('source').split('-').last, r.fetch('group_id'), r.fetch('state')] }
  units=Hash.new { |h, k| h[k]=[] }
  groups.each { |(lang, axis, _, _), indices| units[[lang, axis]] << indices }
  actors=rows.to_h { |r| [r.fetch('id'), r.fetch('world').fetch('facts').map { |f| f.fetch('actor') }.uniq] }
  pools=a::Catalog::LANGUAGES.to_h { |lang| [lang, EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(lang, 'canonical')] }
  name_regex=pools.transform_values { |names| Regexp.union(names.sort_by { |x| -x.length }) }
  renamer=EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: pools)
  old_forbidden=JSON.parse(File.read(File.join(root, 'forbidden.json'))).to_set
  old_v07=JSON.parse(File.read(File.join(a::ROOT_V07, 'forbidden-state-hashes.json'))).to_set
  v07_renamer=EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: pools, forbidden_states: old_v07)
  augmentation=a::Augmentation.new(rows: rows, forbidden: old_forbidden)
  families=Set.new
  materials=Set.new
  streams=[['v07-1337', 1337, 6000, :v07], ['v07-2027', 2027, 6000, :v07], ['v08-reference-prefix', 1337, 1549, :reference], ['v08-coverage-prefix', 1337, 1541, :coverage]]
  streams.each do |name, seed, steps, kind|
   stream_materials=Set.new
   steps.times do |step|
    rng=Random.new(seed+step*1009)
    indices=a::Catalog::LANGUAGES.product(%w[Actor Event]).flat_map { |key| 2.times.flat_map { units.fetch(key).sample(random: rng) } }
    indices.each_slice(16).with_index do |batch, micro|
     batch_seed=(seed+step*1009+micro)%(2**31)
     base=batch.map { |i| examples[i] }
     raw=kind==:v07 ? batch.map { |i| rows[i] } : augmentation.call(base, rng: Random.new(batch_seed^0xE818), coverage: kind==:coverage)
     rng=Random.new(batch_seed^0xA117)
     loop do
      renamed=(kind==:v07 ? v07_renamer : renamer).call(raw.map { |r| EasyAI::Decision::Data::Example.new(r) }, rng: rng)
      forbidden=kind==:v07 ? false : renamed.any? { |r| old_forbidden.include?(a::Catalog.material(r.state)) }
      next if forbidden
      raw.each_slice(4).with_index do |unit, index|
       rendered=renamed.slice(index*4, 4)
       mapping=unit.each_index.to_h { |i| [unit[i].dig('world', 'actor'), rendered[i].question.match(name_regex.fetch(rendered[i].language)).to_s] }
       raise 'Missing renamed identity' if mapping.values.any?(&:empty?)
       facts=unit.first.fetch('world').fetch('facts').map { |fact| fact.merge('actor'=>mapping.fetch(fact.fetch('actor'))) }
       families << a::Catalog.family(unit.first.merge('world'=>unit.first.fetch('world').merge('facts'=>facts)))
       materials << a::Catalog.material(rendered.first.state)
       stream_materials << a::Catalog.material(rendered.first.state)
      end
      break
     end
    end
    if step==1499 && kind != :v07
     arm=kind==:reference ? 'reference' : 'coverage'
     checkpoint=EasyAI::Decision::Checkpoint.resolve(File.join(root, 'arms', "#{arm}-1337", 'choice'))
     actual=JSON.parse(File.read(File.join(checkpoint, 'metadata.json'))).fetch('training').fetch('rendering_coverage').fetch('surface_hashes').keys.to_set
     raise 'Prefix surface replay mismatch' unless actual==stream_materials
     a.write(File.join(destination, "replay-proof-#{arm}.json"), { matched_state_hashes: actual.size, step: 1500, metadata_sha256: a.sha(File.join(checkpoint, 'metadata.json')) })
    end
   end
   puts "#{name}: completed #{steps} steps, #{families.size} cumulative augmented families"
   $stdout.flush
  end
  a.write(File.join(destination, 'replayed-augmented-families.json'), families.to_a.sort)
  a.write(File.join(destination, 'replayed-augmented-materials.json'), materials.to_a.sort)
  counts=a::Catalog::LANGUAGES.to_h do |lang|
   names=pools.fetch(lang)
   events=EasyAI::Decision::Data::FactualContrasts::TRAIN_EVENTS
   available=names.product(events.combination(2).to_a).count { |actor, ev| !families.include?(JSON.generate([lang, ev.map { |e| [actor, e] }.sort])) }
   [lang, available]
  end
  a.write(File.join(destination, 'family-replay-audit.json'), { streams: streams, augmented_families: families.size, augmented_materials: materials.size, available_legacy_event_families: counts })
  puts counts.inspect
end
