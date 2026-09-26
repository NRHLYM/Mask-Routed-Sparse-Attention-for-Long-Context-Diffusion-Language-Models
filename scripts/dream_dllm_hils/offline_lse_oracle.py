#!/usr/bin/env python3
"""Frozen-state exact-LSE counterfactuals from the completed tensor audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dream_dllm_hils.diagnostic_helpers import physical_positions
from dream_dllm_hils.lse_oracle import choose_intervention, evidence_recall, retained_mass
from dream_dllm_hils.lse_calibration import overlap
from scripts.dream_dllm_hils.diagnose_retrieval import sha256


def summarize(records):
    summary = {}
    for phase in ("prefill", "cached_refresh"):
        selected = [r for r in records if r["phase"] == phase]
        variants = {}
        for variant in ("baseline", "select_exact", "fusion_exact"):
            rates, hit_cases, locals_, masses = [], 0, [], []
            for r in selected:
                values = [o[variant]["value"]["recall"] for o in r["observations"]]
                rates.append(float(np.mean([v for v in values if v is not None])))
                hit_cases += any(o[variant]["value"]["hits"] > 0 for o in r["observations"])
                locals_.append(float(np.mean([o[variant]["local_weight"] for o in r["observations"]])))
                masses.append(float(np.mean([o[variant]["remote_mass"] for o in r["observations"]])))
            variants[variant] = dict(case_macro_recall=float(np.mean(rates)), case_recalls=rates,
                                     case_ids=[r["case_id"] for r in selected], any_hit_cases=hit_cases,
                                     local_weight=float(np.mean(locals_)), remote_mass=float(np.mean(masses)))
        delta = np.array(variants["select_exact"]["case_recalls"]) - variants["baseline"]["case_recalls"]
        rng = np.random.default_rng(7)
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(-1)
        summary[phase] = dict(cases=len(selected), variants=variants, selection_delta=float(delta.mean()),
            exploratory_case_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
            improved=int((delta > 0).sum()), worse=int((delta < 0).sum()), tied=int((delta == 0).sum()),
            layers={str(layer): {v: float(np.mean([o[v]["value"]["recall"] for r in selected
                    for o in r["observations"] if o["layer"] == layer]))
                    for v in variants} for layer in [3, 7, 11, 15, 19, 23, 27]})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = {r["case_id"]: r for r in map(json.loads, Path(args.manifest).read_text().splitlines())}
    source = [json.loads(x) for path in sorted(Path(args.source).glob("rank-*.jsonl"))
              for x in path.read_text().splitlines()]
    if len(source) != 28 or len({(r["case_id"], r["phase"]) for r in source}) != 28:
        raise AssertionError("expected completed 14-case paired snapshot audit")
    results = []
    for r in source:
        if r["identity"]["manifest_sha256"] != sha256(args.manifest):
            raise AssertionError("manifest differs from frozen audit")
        case = rows[r["case_id"]]
        values = torch.unique(physical_positions(torch.tensor(sum(case["evidence_values"], [])), 64) // 64)
        facts = torch.unique(physical_positions(torch.tensor(sum(case["evidence_facts"], [])), 64) // 64)
        arrays = np.load(r["tensors"])
        observations = []
        for old in r["observations"]:
            layer = old["layer"]
            get = lambda key: torch.from_numpy(arrays[f"layer{layer}_{key}"])
            estimated, exact, local = get("estimated"), get("exact_token"), get("local_lse")
            dropped, native = ~get("eligible")[:, 0, 0], get("native_indices")
            record = dict(layer=layer)
            for variant in ("baseline", "select_exact", "fusion_exact"):
                idx, gate = choose_intervention(variant, estimated, exact, local, dropped, native)
                record[variant] = dict(value=evidence_recall(idx, dropped, values), fact=evidence_recall(idx, dropped, facts),
                    support_overlap=overlap(idx, native), local_weight=float(gate[..., -1].mean()),
                    remote_mass=retained_mass(exact, dropped, idx))
            if record["fusion_exact"]["value"] != record["baseline"]["value"]:
                raise AssertionError("fixed-support recall changed")
            observations.append(record)
        results.append(dict(case_id=r["case_id"], phase=r["phase"], observations=observations))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "offline.json").write_text(json.dumps(dict(source_identity=source[0]["identity"], records=results), indent=2) + "\n")
    summary = summarize(results)
    (output / "offline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
