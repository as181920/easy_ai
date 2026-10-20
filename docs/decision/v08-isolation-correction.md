# Decision v0.8 — retiring a draft with augmented-family overlap

The first data draft and its early development predictions are **retired evidence**, not the final fresh-panel result. They remain under `runs/decision/event-generalization-v08/`, with their original protocol/source snapshots, all checkpoints, traces and failure logs. The corrected run uses `runs/decision/event-generalization-v08-corrected/`. Event meanings, template partitions, world counts, sampler, losses, seeds, endpoint budget and directional rules are unchanged. The novel-name pool is refreshed to avoid presenting identities from the retired draft as newly reserved identities.

## What failed

The initial preparation excluded historical raw records and structured world families. Training also rejected reserved rendered-state hashes. This missed two cases: actual parent/person-renaming augmentation can create families absent from raw files, and the same facts can appear under another template with a different state hash.

An additional audit canonicalized reserved worlds before checking the saved rendering histories. At committed step 1,500 it found **22 overlapping canonical state hashes** in each coverage arm and **46** in each arm without coverage. Those state-hash counts are not independent-world counts. The four jobs were paused and terminated; their checkpoints and unsuccessful draft evaluation remain preserved. No original-draft final-panel predictions were made. No final model is selected from that draft.

## Replay and replacement

Replay reconstructs actual data selection and person renaming for both v0.7 combined parents (6,000 steps per seed), then the two distinct v0.8 prefix data paths. Event loss does not change data RNG, so reference/event share one prefix stream and coverage/combined the other. The v0.7 replay uses the original per-unit reserved-state rejection; the v0.8 replay uses its original batch rejection. All original source snapshots remain available.

At step 1,500, replayed rendered-state sets match saved checkpoint coverage **exactly**: 11,514 distinct state hashes for reference/event and 11,841 for coverage/combined. Replay also covers the executed discarded tails through steps 1,549 and 1,541 respectively. The resulting exclusion evidence contains **50,946 augmented world families / 81,856 normalized materials**. This supplements historical raw-panel exclusion; it is not a claim about all possible parent pretraining material.

Replacement generation keeps deterministic split/view seeds and excludes retired-panel families plus all replayed parent/prefix families and materials. It uses no prediction scores to choose replacement worlds. This is a data-isolation repair after observations on the retired draft, and must be disclosed as such; it is not an untouched original prospective panel. No replacement-panel prediction precedes the corrected protocol freeze.

Every reserved world is additionally rendered under **all seven frozen forms**, both record orders and all frozen truth assignments for training rejection. This closes cross-template family reuse. Regression tests show that a held expression's canonical rendering, which passes the original literal-state check, is rejected by the strengthened exclusion set.

## Safe prefix reuse and update accounting

Committed step-1,500 model, optimizer and sampler states can be reused only after the replacement exclusions have zero intersection with their rendered histories. The corrected import seals previous/new exclusion signatures and exact checkpoint weights. Resume rejects intervention changes, unsafe overlap and unregistered migrations. Old exclusions remain active, so formerly rejected draws stay rejected. CPU tests verify numerical optimizer continuation under an exclusion-strengthening migration that does not affect the sampled inputs.

The four original traces have 1,523 / 1,541 / 1,538 / 1,549 successful updates for combined / coverage / event / reference. Restoring each at 1,500 repeats **151** updates. Copied prefix traces count their original executions once; appended replay counts again. Original feasibility contributes 12 updates and repeated corrected feasibility may add 12. The worst-case total is therefore **36,175 / 37,000 actual CUDA updates**, including all registered second-seed confirmation. This correction does not authorize extra model search.

Actual resumed CUDA traces now verify **all 151 repeated updates exactly**, including loss components, pre-clipping gradient norm and learning rate. Expected/observed replay counts are 49 reference, 41 coverage, 38 event loss and 23 combined, with no differing trace fields. The numerical replay audit is retained under the corrected root's `isolation-correction/` directory.

The inherited prefixes remain training evidence for the replacement panel because that panel excludes their actual families. Old development curves are not mixed into replacement development curves. The original v0.7 checkpoints are re-evaluated on replacement development before continued optimization. Final choices and the final-panel opening remain separate, sealed stages. Unfavorable results remain publishable outcomes of this round.

The versioned `benchmarks/decision/event_generalization/family_replay.rb` reproduces the exclusion replay from the retired draft root, whose raw training data, exclusions, prefix metadata and traces must remain available. Corrected preparation also requires those audit artifacts. Reconstructing a clean checkout without retained parent/draft artifacts requires recovering them first; the local replay command does not invent missing trained weights or histories.
