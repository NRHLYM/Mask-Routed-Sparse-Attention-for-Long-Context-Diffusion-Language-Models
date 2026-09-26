#!/usr/bin/env python3
"""Case-macro summary of completed paired LSE diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np


def flatten_numeric(record, prefix=""):
    flat = {}
    for key, value in record.items():
        path = prefix + key
        if isinstance(value, dict):
            flat.update(flatten_numeric(value, path + "."))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            flat[path] = value
    return flat


def average(records):
    flat = [flatten_numeric(x) for x in records]
    keys = set.intersection(*(set(x) for x in flat))
    return {key: float(np.mean([x[key] for x in flat])) for key in sorted(keys)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()
    directory = Path(args.directory)
    records = [json.loads(line) for path in sorted(directory.glob("rank-*.jsonl"))
               for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 28 or len({(r["case_id"], r["phase"]) for r in records}) != 28:
        raise AssertionError("expected 14 unique cases x 2 snapshots")
    if any(r["identity"] != records[0]["identity"] for r in records):
        raise AssertionError("mixed provenance")
    result = dict(identity=records[0]["identity"], aggregation="equal case, equal layer, equal query/head rows",
                  snapshots={}, validations={})
    for phase in ("prefill", "cached_refresh"):
        selected = [r for r in records if r["phase"] == phase]
        observations = [o for r in selected for o in r["observations"]]
        result["snapshots"][phase] = dict(cases=len(selected), layer_observations=len(observations),
            query_head_rows=sum(o["route_reference"]["rows"] for o in observations),
            query_kv_group_rows=sum(o["route_group_rank"]["rows"] for o in observations),
            means=average([average(r["observations"]) for r in selected]),
            layers={str(layer): average([o for o in observations if o["layer"] == layer])
                    for layer in sorted({o["layer"] for o in observations})})
    for path in directory.glob("rank-*.validation.json"):
        result["validations"][path.name] = json.loads(path.read_text())
    # Fixed case, layer, query and head chosen independently of the discrepancy.
    sample = next(r for r in records if r["phase"] == "prefill" and r["case_id"] == "paired-L32768-case00")
    arrays = np.load(sample["tensors"])
    layer = 15
    estimate = arrays[f"layer{layer}_estimated"][0, 0, 0]
    truth = arrays[f"layer{layer}_exact_token"][0, 0, 0]
    eligible = arrays[f"layer{layer}_eligible"][0, 0, 0]
    chunks = np.flatnonzero(eligible)
    true_order = chunks[np.argsort(-truth[chunks], kind="stable")]
    estimate_order = chunks[np.argsort(-estimate[chunks], kind="stable")]
    a = int(true_order[0])
    candidates = [int(c) for c in estimate_order if truth[c] < truth[a] and estimate[c] > estimate[a]]
    result["example"] = dict(case_id=sample["case_id"], layer_zero_based=layer,
        query_position=next(o for o in sample["observations"] if o["layer"] == layer)["query_positions"][0],
        kv_head=0, query_head_in_group=0, selected_cherry_pick="fixed row; compare strongest true chunk with its ranking inversion",
        inversion=[])
    for chunk in ([a, candidates[0]] if candidates else [a]):
        result["example"]["inversion"].append(dict(chunk=chunk, estimated=float(estimate[chunk]),
            exact=float(truth[chunk]), true_rank=int(np.flatnonzero(true_order == chunk)[0]) + 1,
            estimated_rank=int(np.flatnonzero(estimate_order == chunk)[0]) + 1))
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    display_keys = ["route_reference.bias", "route_reference.mae", "route_reference.centered_rmse",
        "route_reference.spearman.mean", "route_reference.pair_order_agreement.mean",
        "route_reference.topk_overlap.mean", "route_selector_overlap",
        "token_reference.bias", "token_reference.mae", "token_reference.centered_rmse",
        "token_reference.spearman.mean", "token_reference.pair_order_agreement.mean",
        "token_reference.topk_overlap.mean", "token_selector_overlap",
        "estimated_local_weight.mean", "native_local_weight.mean",
        "route_fixed_support_local_weight.mean", "token_fixed_support_local_weight.mean",
        "token_fixed_support_local_delta.mean", "token_fixed_support_local_abs_delta.mean",
        "fp32_route_reference.mae", "fp32_token_reference.mae", "fp32_selector_native_overlap",
        "native_selector_reconstruction_overlap", "gate_reconstruction_abs_error.mean"]
    print(json.dumps({phase: {k: value["means"][k] for k in display_keys}
                      for phase, value in result["snapshots"].items()}, indent=2))
    print(json.dumps(result["example"], indent=2))


if __name__ == "__main__":
    main()
