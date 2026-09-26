#!/usr/bin/env python3
"""Paired case-macro readout, keeping same-state and propagated effects separate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dream_dllm_hils.lse_calibration import group_priority, top_indices
from dream_dllm_hils.lse_oracle import evidence_recall
from dream_dllm_hils.diagnostic_helpers import physical_positions


def mean(values):
    return float(np.mean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    directory = Path(args.directory)
    offline = json.loads((directory / "offline-summary.json").read_text())
    offline_records = json.loads((directory / "offline.json").read_text())["records"]
    frozen = {(r["case_id"], r["phase"]): r for r in offline_records}
    records = [json.loads(x) for path in sorted((directory / "full").glob("rank-*.jsonl"))
               for x in path.read_text().splitlines()]
    if len(records) != 84 or len({(r["case_id"], r["variant"], r["phase"]) for r in records}) != 84:
        raise AssertionError("expected 14 cases x 3 variants x 2 snapshots")
    if any(r["identity"] != records[0]["identity"] for r in records):
        raise AssertionError("mixed live-run provenance")
    summary = dict(identity=records[0]["identity"], frozen_state=offline, propagated={}, validations={},
                   scope="oracle interventions only on 32 active answer predictors; statistics on matched 16 predictors")
    for phase in ("prefill", "cached_refresh"):
        summary["propagated"][phase] = {}
        for variant in ("baseline", "select_exact", "fusion_exact"):
            group = [r for r in records if r["phase"] == phase and r["variant"] == variant]
            ordered = sorted(group, key=lambda r: r["case_id"])
            rates = [mean([o["actual_value"]["recall"] for o in r["observations"]]) for r in ordered]
            if variant == "fusion_exact":
                if not all(o["support_matches_frozen_baseline"] for r in group for o in r["observations"]):
                    raise AssertionError("fusion support not frozen")
                for r in group:
                    baseline = next(b for b in records if b["variant"] == "baseline" and
                                    b["phase"] == phase and b["case_id"] == r["case_id"])
                    if any(a["actual_value"] != b["actual_value"] for a, b in zip(r["observations"], baseline["observations"])):
                        raise AssertionError("fusion-only evidence recall changed")
            summary["propagated"][phase][variant] = dict(cases=len(group), case_ids=[r["case_id"] for r in ordered],
                case_macro_recall=mean(rates), case_recalls=rates,
                fact_chunk_recall=mean([mean([o["actual_fact"]["recall"] for o in r["observations"]]) for r in group]),
                local_weight=mean([mean([o["actual_local_weight"] for o in r["observations"]]) for r in group]),
                remote_mass=mean([mean([o["actual_remote_mass"] for o in r["observations"]]) for r in group]),
                gold_nll=mean([r["gold_answer_nll"] for r in group]),
                gold_nll_cases=[r["gold_answer_nll"] for r in ordered],
                any_hit_cases=sum(any(o["actual_value"]["hits"] > 0 for o in r["observations"]) for r in group),
                logits_relative_l2=mean([r["logits_vs_baseline"]["relative_l2"] for r in group]),
                baseline_support_overlap=mean([mean([o["baseline_support_overlap"] for o in r["observations"]]) for r in group]))
        f = offline[phase]["variants"]
        delta = np.array(f["select_exact"]["case_recalls"]) - f["baseline"]["case_recalls"]
        largest = int(np.argmax(np.abs(delta)))
        keep = np.arange(len(delta)) != largest
        summary["frozen_state"][phase]["outlier_sensitivity"] = dict(
            omitted_case=f["baseline"]["case_ids"][largest], omitted_delta=float(delta[largest]),
            baseline_remaining_mean=mean(np.array(f["baseline"]["case_recalls"])[keep]),
            oracle_remaining_mean=mean(np.array(f["select_exact"]["case_recalls"])[keep]),
            note="descriptive sensitivity, not replacement of the primary 14-case estimate")
    mismatches = []
    for r in records:
        if r["variant"] == "baseline":
            for a, b in zip(r["observations"], frozen[(r["case_id"], r["phase"])]["observations"]):
                if a["actual_value"]["hits"] != b["baseline"]["value"]["hits"]:
                    mismatches.append(dict(case_id=r["case_id"], phase=r["phase"], layer=a["layer"],
                                           live=a["actual_value"]["hits"], frozen=b["baseline"]["value"]["hits"]))
    summary["baseline_replay_hit_mismatches"] = mismatches
    for path in (directory / "full").glob("*.validation.json"):
        summary["validations"][path.name] = json.loads(path.read_text())
    # Check whether the poor teacher recall is only an artifact of GQA normalization.
    cases = {r["case_id"]: r for r in map(json.loads, Path(args.manifest).read_text().splitlines())}
    calibration = [json.loads(x) for path in sorted(Path(args.calibration).glob("rank-*.jsonl"))
                   for x in path.read_text().splitlines()]
    checks = []
    for r in calibration:
        value = torch.unique(physical_positions(torch.tensor(sum(cases[r["case_id"]]["evidence_values"], [])), 64) // 64)
        arrays = np.load(r["tensors"])
        rates = {key: [] for key in ("native_per_head", "exact_per_head", "exact_gqa_early_mask")}
        for layer in (3, 7, 11, 15, 19, 23, 27):
            get = lambda key: torch.from_numpy(arrays[f"layer{layer}_{key}"])
            estimated, exact, local = get("estimated"), get("exact_token"), get("local_lse")
            dropped = ~get("eligible")[:, 0, 0]
            for name, z in (("native_per_head", estimated), ("exact_per_head", exact)):
                idx = top_indices(z.masked_fill(dropped[:, None, None], -torch.inf), 16).flatten(1, 2)
                rates[name].append(evidence_recall(idx, dropped, value)["recall"])
            idx = top_indices(group_priority(exact, local, dropped, eligible_denominator=True), 16)
            rates["exact_gqa_early_mask"].append(evidence_recall(idx, dropped, value)["recall"])
        checks.append(dict(case_id=r["case_id"], phase=r["phase"], **{k: mean(v) for k, v in rates.items()}))
    summary["teacher_aggregation_checks"] = {phase: {k: mean([r[k] for r in checks if r["phase"] == phase])
        for k in ("native_per_head", "exact_per_head", "exact_gqa_early_mask")} for phase in ("prefill", "cached_refresh")}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"propagated": summary["propagated"], "teacher_checks": summary["teacher_aggregation_checks"],
        "baseline_replay_hit_mismatches": mismatches,
        "outlier_sensitivity": {p: v["outlier_sensitivity"] for p, v in summary["frozen_state"].items()}}, indent=2))


if __name__ == "__main__":
    main()
