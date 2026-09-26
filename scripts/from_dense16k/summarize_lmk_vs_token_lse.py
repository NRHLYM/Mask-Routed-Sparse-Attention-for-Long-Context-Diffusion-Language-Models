#!/usr/bin/env python3
"""Aggregate LMK vs token-LSE jsonl into a diagnosis table."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from dream_dllm_hils.lmk_token_lse_stats import classify_failure


KEYS = [
    "support_frac",
    "mean_lmk_rank",
    "rank_lmk_selected",
    "rank_lmk_all",
    "rank_fusion_selected",
    "mean_token_rank",
    "mean_evidence_rank",
    "corr_lmk_token",
    "corr_fusion_lmk",
    "corr_fusion_token",
    "mean_correct_fusion",
    "mean_fusion_total",
]


def _mean(values: list[float]) -> float:
    finite = [x for x in values if x is not None and math.isfinite(x)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.jsonl:
        rows.extend(load_jsonl(Path(path)))

    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_examples: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        task = row["task"]
        for layer, payload in row["layers"].items():
            key = (task, str(layer))
            n_examples[key] += 1
            for item in payload["items"]:
                for name in KEYS:
                    buckets[key][name].append(float(item.get(name, float("nan"))))

    summary = []
    for (task, layer), metrics in sorted(buckets.items()):
        rec = {
            "task": task,
            "layer": int(layer),
            "n_examples": n_examples[(task, layer)],
        }
        for name in KEYS:
            rec[name] = _mean(metrics[name])
        rec["verdict"] = classify_failure(
            support_frac=rec["support_frac"],
            mean_lmk_rank=rec["mean_lmk_rank"],
            mean_token_rank=rec["mean_token_rank"],
            corr_lmk_token=rec["corr_lmk_token"],
            corr_fusion_lmk=rec["corr_fusion_lmk"],
        )
        summary.append(rec)

    task_roll = defaultdict(lambda: defaultdict(list))
    for rec in summary:
        for name in KEYS:
            task_roll[rec["task"]][name].append(rec[name])
    rolled = []
    for task, metrics in sorted(task_roll.items()):
        rec = {"task": task, "layer": "all_hils"}
        for name in KEYS:
            rec[name] = _mean(metrics[name])
        rec["verdict"] = classify_failure(
            support_frac=rec["support_frac"],
            mean_lmk_rank=rec["mean_lmk_rank"],
            mean_token_rank=rec["mean_token_rank"],
            corr_lmk_token=rec["corr_lmk_token"],
            corr_fusion_lmk=rec["corr_fusion_lmk"],
        )
        rolled.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"per_layer": summary, "per_task": rolled, "n_rows": len(rows)}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"rows={len(rows)}")
    print(
        f"{'task':<12}{'layer':<12}{'sup':>6}{'lmkAll':>8}{'lmkSel':>8}{'fusSel':>8}{'tokR':>7}{'rWL':>7}  verdict"
    )

    def fmt(rec, key, width):
        value = rec.get(key, float("nan"))
        if value is None or not math.isfinite(value):
            return f"{'nan':>{width}}"
        return f"{value:{width}.2f}"

    for rec in rolled + summary:
        print(
            f"{rec['task']:<12}{str(rec['layer']):<12}"
            f"{fmt(rec, 'support_frac', 6)}"
            f"{fmt(rec, 'rank_lmk_all', 8)}"
            f"{fmt(rec, 'rank_lmk_selected', 8)}"
            f"{fmt(rec, 'rank_fusion_selected', 8)}"
            f"{fmt(rec, 'mean_token_rank', 7)}"
            f"{fmt(rec, 'corr_fusion_lmk', 7)}"
            f"  {rec['verdict']}"
        )


if __name__ == "__main__":
    main()
