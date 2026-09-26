#!/usr/bin/env python3
"""Case-paired summary of the independent dense reference and frozen sparse audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dream_dllm_hils.dense_reference import LAYERS, local_chunks, summarize_exact
from dream_dllm_hils.diagnostic_helpers import physical_positions
from dream_dllm_hils.lse_oracle import evidence_recall
from scripts.dream_dllm_hils.diagnose_retrieval import sha256


def arm_summary(records):
    records = sorted(records, key=lambda r: r["case_id"])
    case_recalls = [float(np.mean([o["value"]["recall"] for o in r["observations"]])) for r in records]
    result = dict(cases=len(records), case_macro_recall=float(np.mean(case_recalls)),
                  case_ids=[r["case_id"] for r in records], case_recalls=case_recalls,
                  any_hit_cases=sum(any(o["value"]["any_hit"] for o in r["observations"]) for r in records),
                  layers={str(i): float(np.mean([o["value"]["recall"] for r in records
                                                for o in r["observations"] if o["layer"] == i])) for i in LAYERS})
    counts = [o["value"] for r in records for o in r["observations"]]
    result["hits"] = sum(v["hits"] for v in counts)
    result["units"] = sum(v["units"] for v in counts)
    for metric in ("early_value", "per_head_value", "fact"):
        if metric in records[0]["observations"][0]:
            result[metric + "_recall"] = float(np.mean([
                np.mean([o[metric]["recall"] for o in r["observations"]]) for r in records]))
    for metric in ("evidence_all_attention_mass", "evidence_remote_attention_mass", "retained_remote_mass"):
        if metric in records[0]["observations"][0]:
            result[metric] = float(np.mean([np.mean([o[metric] for o in r["observations"]]) for r in records]))
    if "gold_nll" in records[0]:
        result["gold_nll"] = float(np.mean([r["gold_nll"] for r in records]))
    return result


def paired_delta(left, right):
    if left["case_ids"] != right["case_ids"]:
        raise AssertionError("unpaired case order")
    d = np.array(right["case_recalls"]) - np.array(left["case_recalls"])
    rng = np.random.default_rng(7)
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(-1)
    return dict(delta=float(d.mean()), exploratory_case_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
                improved=int((d > 0).sum()), worse=int((d < 0).sum()), tied=int((d == 0).sum()),
                case_deltas=d.tolist())


def main(args):
    rows = {r["case_id"]: r for r in map(json.loads, Path(args.manifest).read_text().splitlines())
            if r["physical_length"] == 32768}
    dense = [r for p in sorted(Path(args.dense).glob("*.jsonl")) for r in map(json.loads, p.read_text().splitlines())]
    if len(dense) != 28 or len({(r["case_id"], r["arm"]) for r in dense}) != 28:
        raise AssertionError("expected exactly 14 paired cases in both dense arms")
    for case_id in rows:
        pair = [r for r in dense if r["case_id"] == case_id]
        if len(pair) != 2 or {r["arm"] for r in pair} != {"base_dense", "finetuned_dense"}:
            raise AssertionError("missing paired case")
        for field in ("input_sha256", "positions_sha256", "key_mask_sha256"):
            if pair[0][field] != pair[1][field]:
                raise AssertionError("dense inputs differ: " + field)
        for r in pair:
            if r["identity"]["manifest_sha256"] != sha256(args.manifest):
                raise AssertionError("manifest mismatch")
            if r["layers_executed"] != list(range(28)):
                raise AssertionError("missing dense layer")
    source = [r for p in sorted(Path(args.source).glob("rank-*.jsonl"))
              for r in map(json.loads, p.read_text().splitlines()) if r["phase"] == "prefill"]
    sparse, native, recomputed = [], [], []
    for old in source:
        case = rows[old["case_id"]]
        values = torch.unique(physical_positions(torch.tensor(sum(case["evidence_values"], [])), 64) // 64)
        facts = torch.unique(physical_positions(torch.tensor(sum(case["evidence_facts"], [])), 64) // 64)
        records, native_records, recomputed_records = [], [], []
        with np.load(old["tensors"]) as arrays:
            for o in old["observations"]:
                layer = o["layer"]
                get = lambda name: torch.from_numpy(arrays[f"layer{layer}_{name}"])
                exact, local, dropped = get("exact_token"), get("local_lse"), ~get("eligible")[:, 0, 0]
                stats, _ = summarize_exact(exact, local, dropped, values, facts)
                stats["layer"] = layer
                records.append(stats)
                native_records.append(dict(layer=layer, value=evidence_recall(get("native_indices"), dropped, values)))
                lm = local_chunks(torch.tensor(o["query_positions"]), 32768)
                local_exact = torch.logsumexp(exact.masked_fill(~lm[:, None, None], -torch.inf), -1)
                check, _ = summarize_exact(exact, local_exact, dropped, values, facts)
                check["layer"] = layer
                recomputed_records.append(check)
        sparse.append(dict(case_id=old["case_id"], observations=records))
        native.append(dict(case_id=old["case_id"], observations=native_records))
        recomputed.append(dict(case_id=old["case_id"], observations=recomputed_records))
    if len(sparse) != 14:
        raise AssertionError("incomplete sparse reference")
    arms = {arm: arm_summary([r for r in dense if r["arm"] == arm])
            for arm in ("base_dense", "finetuned_dense")}
    arms.update(finetuned_sparse_exact=arm_summary(sparse), finetuned_sparse_native=arm_summary(native),
                finetuned_sparse_exact_recomputed_local=arm_summary(recomputed))
    result = dict(arms=arms, comparisons={
        "base_dense_to_finetuned_dense": paired_delta(arms["base_dense"], arms["finetuned_dense"]),
        "finetuned_dense_to_finetuned_sparse_exact": paired_delta(arms["finetuned_dense"], arms["finetuned_sparse_exact"]),
        "base_dense_to_finetuned_sparse_exact": paired_delta(arms["base_dense"], arms["finetuned_sparse_exact"])},
        checks=dict(paired_inputs=True, independent_base_no_adapters=True, all28layers_dense=True,
                    manifest_sha256=sha256(args.manifest)),
        interpretation="dense retrieval diagnostics, not generated answer correctness; case-level pairing; no independent-unit significance",
        limitations=["Initial all-mask prefill only", "YaRN factor4/original8192 held fixed, not native2k-to32k correction",
                     "Fine-tuned dense is a counterfactual inference topology, not a densely trained model",
                     "Dense-vs-sparse difference bundles sparse attention, Q-Cal routing, LMK and fusion; not a single component cause",
                     "Placeholder keys excluded and original physical positions preserved"])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
