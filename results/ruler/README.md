# RULER probe scores (not training data)

Goldspan S-N / MK-MQ / VT. Each `metrics.json` is 100 examples at one length.
`rank-0.jsonl` is the raw predictions. Worker logs are omitted.

Ablations are keep-train YaRN factor=8 (`ruler_probes_goldspan_yarn8fixed`).
Main-table MRSA / dense / NSA / DSA / hybrid dirs named `ruler_probes_goldspan` use YaRN = L/2048.
DSA and hybrid also have a yarn8fixed copy where that eval was run.
