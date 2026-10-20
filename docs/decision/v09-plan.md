# Decision v0.9 — independent wording coverage and retention

Status: authorized 2026-10-20, prospective plan before v0.9 prediction/training. The completed [v0.8 experiments](v08.md) show repeated new-event gains but failed complete seed confirmation and severe same-world template sensitivity. The user accepted completing the next round. v0.8 is committed as `6b78b4d`; preserve its data, sources, weights, failures and documentation. This round does not authorize a push or an automatic commit of v0.9.

## Fixed model and controlled factors

Start from the exact own v0.7 combined checkpoints at seeds 1337/2027, not the expression-regressed v0.8 branch. All arms keep 6,627,841 parameters, separate/RoPE encoding, own tokenizer, person renaming, CE plus `.5` actor-pair loss, LR `.0003`, fresh AdamW state, no dropout/warmup, effective batch32/micro16/accumulation2 and 6,000 updates. There is no new event auxiliary loss, model-size search, natural QA, unknown objective, calibration or release promotion.

The original8,192 binary training rows and balanced four-query sampler are identical. Independent seeded RNG streams draw event mappings and forms before applying the intervention, preserving event choices across event-covered arms and form choices across the matched expression-covered arms. Each language/axis bucket samples two complete units; transformation applies to the second unit. A whole unit shares its fact mapping and retains all four labels/questions. Training renaming updates semantic metadata along with text, rejects reserved canonical families before any rendering and records actual families, states, axis/form and event exposure. Metadata never enters model input.

| Arm | New events | New wording | Canonical-form replay |
| --- | --- | --- | --- |
| reference | None | None | 100% |
| events | 50% units | None | 100% |
| expressions | None | 50% units | 50% |
| both | 50% units | 50% units | 50% |
| no_replay | 50% units | 100% units | 0% |

The first four arms form an event × wording factorial. The fifth compares canonical replay against additional wording exposure at fixed original-row visits and event coverage. It changes the old/new form exposure tradeoff; it does not isolate replay from new-form exposure at equal FLOPs or equal convergence. All five are prospectively eligible for the same selection rule. Report every arm and the direct both/no_replay contrast even if neither qualifies.

Per-unit mappings can differ across sampled units from the same original world. Inherited cross-unit `fact_flip` IDs in augmented training are bookkeeping, not semantic counterfactual families; neither the loss nor selection uses those augmented cross-unit metrics. Complete held worlds and the globally mapped fixed probe preserve true counterfactual families.

## Fresh allocation and fully crossed evaluation

Add24 event entries with8 training,8 development and8 final entries. Entries are reviewed explicit positive/negative event phrases, not proof of entirely novel real-world concepts or unseen token fragments. Reuse old canonical form0; train wording coverage on forms1/2/7/8. Reserve forms9/10 for development and11/12 for final, with no expression-ID overlap across these pools. Forms3–6 remain historical v0.8 regression wording. Neither new event IDs nor more prefixes establish broad language understanding.

Reserve unordered actor/event fact families before all truth assignments, queries, polarities, record orders and renderings. Exclude all historical raw-panel families/materials plus the reproduced actual v0.7 parent augmentation evidence retained by v0.8; novel identities must be absent from historical structured fact metadata. Later v0.8 branch augmentations do not initialize v0.9 and are not fully replayed: the parent-isolation claim covers the actual v0.7 parents and the audited historical raw panels, not every possible unrelated-branch training rendering. Preserve the evidence hashes and scope. Existing v0.8 worlds are regression data, never new acceptance.

Five views separate combinations, actors, events, expressions and composed shifts. Development allocates32 worlds per language/view; final64. Actor/event axes divide worlds equally. **Every expression/composed world is rendered under both held forms**, so the same family receives both templates on either axis. Canonical views use form0. All four truth assignments, both record orders and both claim polarities stay in one split/family. Forms get distinct row/group/pair IDs and share an explicit `family_id`.

This yields **14,336 development /28,672 final binary rows**. Per axis/language/form there are128 development /256 final mixed-truth pairs sharing16/32 worlds. Merging both held forms doubles pair counts without adding worlds. Independently reconstruct overall and every per-form pair score, and report world-level paired changes. Frozen generated data and actual tokenizer/collator must have no opposite-label input collisions or truncation. Reject finite allocation exhaustion rather than repeat worlds.

Training rejects every reserved family regardless of template, plus all13 frozen forms' state renderings and inherited exact-state exclusions. Reserve v0.7 evaluation and v0.8 final families for regression protection too. Record and audit executed semantic families directly; a state hash alone cannot establish family isolation. Full-cap256/128 CUDA probes validate micro16/accumulation2 under4,096MiB per-process, with no silent CPU fallback. Tests use CPU and deterministic mathematical assertions.

The fixed original and globally event-mapped/form7 training probes are diagnostics every500 updates. The latter maps whole world families consistently, retaining valid fact-flip pairs. A fixed first-four-world subset supplies development curves every1,000 updates; only full6,000-step endpoints select the recipe. Evaluate original full training and observed v0.8 regression views at endpoints. Historical expression/composed regression includes both original and crossed templates on the same families.

## Selection, confirmation and final opening

The target cells are both actor/event full-pair scores across new-event canonical form0 and both new-expression forms, in each language: **12 cells**. A candidate qualifies only if the minimum target cell improves by at least5 percentage points over the matched reference, no individual target cell falls by more than5 points, no basic-view/language/form balanced accuracy falls by more than3 points, and no v0.8 regression language/axis pair score falls by more than5 points or balanced accuracy by more than3 points. Composed shift is a separately reported stress diagnostic.

Rank qualifying arms by minimum target cell, mean target cell, worst basic-view balanced accuracy, then negative NLL. All original-row visit arrays must match. Freeze selection after all five primary endpoints. If qualifying, run only the selected candidate and matched reference at seed2027 for6,000 updates each, with the same rule. Preserve failed confirmation; never select an alternative afterward. Training95% balanced /85% group scores remain advisory diagnostics, not early stops or production gates.

Before any final predictions, seal both original parents and all five primary endpoints; if confirmation is triggered also include the two confirmation endpoints, even if confirmation fails. Thus the fixed final panel has7 or9 checkpoints. Open once; do not optimize after opening. Reporting reuses saved predictions with integrity hashes. CPU/CUDA and public candidate-order/chunk checks use the first four whole worlds per language/view and every held form:256 rows for canonical views,512 for expression/composed views. Label any device differences and this subset scope honestly; full CUDA predictions have independent per-form pair reconstruction.

## Budget and deliverables

Five primary arms ×6,000 =30,000 updates; conditional confirmation adds12,000. Five shape probes add15. **Maximum planned42,015 actual CUDA updates; hard ceiling43,000** counts all successful physical/replayed optimizer updates. No additional searches or unregistered extensions. File locks protect each arm, exact optimizer/augmentation resume is tested, and all failed invocations and source amendments remain preserved. Freeze scientific source/config/data/parent/plan hashes before predictions; only read-only presentation/audit/reporting may evolve with final helper hashes recorded.

Complete core correctness tests, full repository checks, all registered experiments, fixed final evaluation, isolated-family/exposure audits, interface/device audits, paired error analysis, reviewed charts, machine-readable results and loading/reproduction instructions. Update both READMEs, handover and retrospective. The conclusion may be positive, partial or bounded negative; v0.1 remains the delivered preview. Existing v0.9 panels become observed regression after use and cannot support a later fresh-selection claim.
