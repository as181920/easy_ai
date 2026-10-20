# Decision v0.7 — runtime corrections and preserved evidence

The [original plan](v07-plan.md), protocol and source snapshots remain immutable under the ignored local root `runs/decision/generalization-v07/`. Corrections are explicit, hashed amendments rather than replacements of earlier evidence. None changes the model, numerical training loop, losses, sampler, data, optimizer hyperparameters or direction criterion.

## Namespace collision and replay

The new benchmark initially reused the historical `generalization.rb` entrypoint and test filename. The historical files are restored byte-for-byte from the existing commit; v0.7 now uses `binding_generalization.rb`, `DecisionBindingGeneralization` and a separate test class. `namespace-correction/amendment.json` records the corrected source snapshot and a comparison showing identical trainer/experiment bodies after namespace replacement.

CE/1337 was interrupted after 825 successful updates and resumed from committed checkpoint 750. All 75 replayed updates are counted in the actual-update ledger. Their trace fields match the interrupted execution exactly, including zero maximum numerical difference. Original logs, trace snapshots, runtime sources and the replay audit remain under `namespace-correction/`.

Independent primary workers were subsequently scheduled concurrently after all memory probes passed. Per-arm file locks protect output, and completed endpoints are preserved. Their elapsed intervals are not exclusive GPU throughput measurements.

## Reference confirmation

The [reference-confirmation clarification](v07-exposure-confirmation.md) always retains CE/2027, including when an intervention fails to beat CE. Its amendment chains after the namespace correction. This keeps the training-duration question testable without changing the original maximum allocation or numerical training body.

A controller child that had started before this amendment retained the earlier source verifier while waiting on another worker's lock. When that worker finished, the stale verifier rejected the newly amended source hash. The controller failure and original log are preserved; the completed training endpoint was unaffected. A fresh controller reads the amended source chain. The same check can reject any waiting child loaded before a later amendment; restart that controller after preserving its failure rather than suppress the integrity check.

## Selection integrity lookup

The frozen selection JSON stores result hashes under variant names (`ce`, `rename`, `pair`, `combined`). Its initial reader incorrectly interpreted those keys as file paths, so the first combined/2027 invocation failed before any optimizer update. The correction resolves each key to `arms/VARIANT-1337/result-6000.json` and checks the exact registered arm set.

`selection-integrity/amendment.json` chains after the reference amendment and retains before/after sources. The original selection and its metrics remain unchanged. Deterministic regression tests cover successful loading, a changed result file and a changed selection file. Failure JSON and stderr remain available. Subsequent confirmation uses the same frozen recipe and seed.

Read-only audit, plotting and conclusion helpers were added separately. They produce independent pair counts, raw CPU/CUDA predictions, public candidate-order checks and compact reporting; they perform no optimizer updates. The final report verifies the original protocol plus the chained source amendments and counts every executed training trace entry.

## Final reporting snapshot

The selection correction's full source snapshot also included the then-current presentation helper. Later read-only reporting improvements therefore correctly failed source verification on two audit attempts. Those failures performed no training. `reporting-integrity/amendment.json` explicitly freezes the finished report/conclusion helpers and CLI, and extends the verifier's amendment chain. It checks that changes are limited to reporting, CLI and amendment recognition; trainer, numerical experiment and library sources remain identical to the preceding manifest.

The conclusion additionally verifies every file in all five archived source snapshots against its original manifest: initial protocol, namespace runtime, reference policy, selection correction and final reporting. Failed audit stderr/JSON are retained; successful audits are rerun using the final manifest. No integrity check is disabled.
