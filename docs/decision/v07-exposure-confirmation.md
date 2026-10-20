# Decision v0.7 — reference confirmation clarification

Registered during primary training on 2026-10-20, after observing CE's canonical/training learning curve and before the four-arm endpoint selection. The original plan and namespace-correction records remain unchanged. This is an explicit, data-informed clarification, not an initially preregistered rule.

The round asks whether earlier training was insufficient as well as whether the new interventions beat CE. Those are separate questions. Requiring an intervention to beat CE before permitting a CE reference replication could leave the training-sufficiency result unconfirmed even when longer CE itself helps.

Therefore run the **CE/128 reference at seed 2027 for 6,000 updates regardless of intervention selection**. Keep its parent, data, optimizer, objective, sampler and checks identical. An improving intervention still receives the originally planned seed-2027 confirmation against that CE reference. The reference remains one run; file locks prevent the controller and direct invocation from duplicating it.

The maximum planned own-model work remains 42,000 unique steps: four primary arms (24k), optional CE extension (6k), CE reference confirmation (6k), and optional selected-intervention confirmation (6k). The 42,500 hard ceiling still counts every probe and replay. Without an improving intervention, the planned frozen-pretrained diagnostic remains available; report any successful CE learning separately rather than describing the whole round as negative.

This changes only permission to start the CE reference; numerical training code and all selection thresholds are unchanged. Preserve original/runtime source hashes and verify the training body is byte-identical. Reference performance is observed development, not a newly reserved acceptance or a factual release.
