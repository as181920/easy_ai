# Decision v0.6 — audit of near-zero actor-pair correctness

The user requested a correctness investigation because 0%–3.125% appears worse than guessing. That concern is valid: independent binary guesses have 50% expected row accuracy and **25% expected two-row pair correctness**. Near-zero pair correctness needs an explanation; merely saying that the metric is stricter is insufficient.

The reported slice contains **mixed-truth, same-event, two-actor questions on held-out actors**, not every question involving a new person. Each pair keeps the record and assertion polarity fixed, changes the queried actor, and has opposite correct answers. A pair counts as correct only when both answers are correct. There are 32 pairs per language in this observed development slice: eight actor worlds, two mixed truth assignments and two assertion polarities. These pairs share worlds and are not 32 independent family samples.

Fresh CPU inference from both 4,000-update checkpoints gives:

| Capacity | Language | Correct rows / 64 | Zero / one / two correct answers per pair | Pair correctness |
| --- | --- | ---: | ---: | ---: |
| Base | English | 30 | 2 / 30 / 0 | 0/32 = 0% |
| Base | Chinese | 32 | 0 / 32 / 0 | 0/32 = 0% |
| Large | English | 32 | 0 / 32 / 0 | 0/32 = 0% |
| Large | Chinese | 31 | 2 / 29 / 1 | 1/32 = 3.125% |

These are correctness rates, not error rates. For base/Chinese and large/English, every pair gets the same predicted answer on both questions. Exactly one answer is correct, yielding 50% row accuracy and 0% pair correctness. Independent random guessing really does have higher expected pair correctness. The observations show a correlated failure, without uniquely identifying its architectural cause.

For example, the base model sees:

```text
Record: Kendall Reed did not close the window. Parker Reed closed the window.
Claim 1: Kendall Reed closed the window.  Gold: no.  Predicted: no (50.593%).
Claim 2: Parker Reed closed the window.   Gold: yes. Predicted: no (50.496%).
```

The model changes its scores slightly but chooses `no` for both claims. One row is right; the pair is wrong. This is an observed development example, not an independent test or a rule used by the model.

The [audit implementation](../../benchmarks/decision/scaling/binding_audit.rb) checks the following:

- Independently reconstruct pairs from rendered state and assertion polarity, without using the evaluator's contrast-group IDs. Count zero, one and two correct answers explicitly. Require two opposite-label rows per pair.
- Compare those counts with the existing evaluator and the recorded CUDA results. Both capacities agree exactly on actor-pair correctness.
- Validate generated labels against explicit facts and validate the rendered state/question against those facts. Check actual tokenized inputs for opposite-label collisions and truncation.
- Compare batched evaluation with the public predictor on eight bilingual examples per model, reversing candidate order and using candidate chunks of one. Predictions agree by candidate ID.
- Test perfect, fully inverted and constant-answer predictions against the actual evaluator with reversed candidate ordering. Exhaustively enumerate binary guesses to recover the 25% random-pair expectation.
- Exercise the real candidate-training shuffle and collator, check that target tensor indexes still identify the intended candidate ID, and independently calculate its cross-entropy from model scores. This tests label/loss alignment without requiring convergence.

The input audit covers **34,304 actual tokenized examples** (32,768 training, 1,024 canonical development and 512 novel-actor development), with zero opposite-label collisions and zero truncations. The full CPU suite passes **269 tests / 6,457 assertions**, including the new metric and training-label tests.

No correctness/error inversion or candidate-label mismatch was found in these checks. This is evidence about the audited paths, not a proof that every implementation detail is correct. Low pair correctness remains an unresolved model behavior; lowering the development retention floor from 90% to 85% does not change these held-out results.

Reproduce from the project root, with local experiment artifacts:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby benchmarks/decision/scaling.rb --phase audit-binding --profile base
bundle exec ruby benchmarks/decision/scaling.rb --phase audit-binding --profile large
bundle exec ruby benchmarks/decision/scaling.rb --phase audit-binding-inputs
bundle exec ruby -Itest test/easy_ai/decision/scaling_test.rb
```

Raw logits, prediction IDs, records, targets, counts, tokenizer checks and weights/prediction hashes are retained under `runs/decision/scaling-v06/binding-audit/`. These audits use observed development and do not open reserved acceptance or execute optimizer updates.
