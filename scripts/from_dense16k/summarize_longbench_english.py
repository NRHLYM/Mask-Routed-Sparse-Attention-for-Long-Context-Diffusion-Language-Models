#!/usr/bin/env python3
"""Summarize LongBench-v1 English-16 only (skip zh tasks)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dream_dllm_hils.longbench_eval import LONGBENCH_EN_TASKS


def count_jsonl(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    task_metrics: dict[str, dict[str, object]] = {}
    for task in LONGBENCH_EN_TASKS:
        metrics_path = output_root / task / "metrics.json"
        data_path = data_root / f"{task}.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics for {task}: {metrics_path}")
        if not data_path.is_file():
            raise FileNotFoundError(f"missing LongBench data for {task}: {data_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_examples = count_jsonl(data_path)
        if str(metrics.get("task")) != task:
            raise ValueError(f"{task}: metrics task mismatch: {metrics.get('task')}")
        if int(metrics.get("examples", -1)) != expected_examples:
            raise ValueError(
                f"{task}: expected {expected_examples} examples, "
                f"got {metrics.get('examples')}"
            )
        if int(metrics.get("fallback_max", -1)) != 0:
            raise ValueError(f"{task}: nonzero kernel fallback")
        task_metrics[task] = {
            "examples": expected_examples,
            "metric": str(metrics["metric"]),
            "score": float(metrics["score_avg"]),
            "mean_seconds": float(metrics["mean_seconds"]),
            "p50_seconds": float(metrics["p50_seconds"]),
            "fallback_max": int(metrics["fallback_max"]),
        }
    english_scores = [task_metrics[task]["score"] for task in LONGBENCH_EN_TASKS]
    payload: dict[str, object] = {
        "suite": "LongBench-v1-english16",
        "english_task_count": len(LONGBENCH_EN_TASKS),
        "all_task_count": len(LONGBENCH_EN_TASKS),
        "english_macro_average": sum(english_scores) / len(english_scores),
        "all_macro_average": sum(english_scores) / len(english_scores),
        "total_examples": sum(
            int(task_metrics[task]["examples"]) for task in LONGBENCH_EN_TASKS
        ),
        "skipped_zh_tasks": [
            "multifieldqa_zh",
            "dureader",
            "vcsum",
            "lsht",
            "passage_retrieval_zh",
        ],
        "tasks": task_metrics,
    }
    write_json_atomic(output_root / "suite_metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
