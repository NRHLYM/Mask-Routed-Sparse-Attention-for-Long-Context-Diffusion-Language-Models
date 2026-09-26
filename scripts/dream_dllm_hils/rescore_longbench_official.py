#!/usr/bin/env python3
"""Rescore LongBench Fast-dLLM shards with official task metrics (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dream_dllm_hils.longbench_eval import LONGBENCH_ALL_TASKS, longbench_score


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def rescore_task(
    *,
    output_root: Path,
    data_root: Path,
    task: str,
    inplace: bool,
) -> dict[str, object] | None:
    task_dir = output_root / task
    merged_path = task_dir / "merged.jsonl"
    metrics_path = task_dir / "metrics.json"
    if not merged_path.is_file() or not metrics_path.is_file():
        return None
    examples = _read_jsonl(data_root / f"{task}.jsonl")
    records = _read_jsonl(merged_path)
    metrics_old = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_out: list[str] = []
    for record in records:
        index = int(record["index"])
        example = examples[index]
        answers = [
            str(answer) for answer in example.get("answers", record.get("answers", []))
        ]
        raw_classes = example.get("all_classes")
        all_classes = (
            [str(label) for label in raw_classes]
            if isinstance(raw_classes, list)
            else None
        )
        score, metric = longbench_score(
            task,
            str(record["prediction"]),
            answers,
            all_classes=all_classes,
        )
        record["score"] = score
        record["metric"] = metric
        record["answers"] = answers
        metrics_out.append(metric)
    if len(set(metrics_out)) != 1:
        raise ValueError(f"{task}: mixed metrics {set(metrics_out)}")
    metric = metrics_out[0]
    score_avg = 100.0 * sum(float(record["score"]) for record in records) / len(records)
    payload = {
        "task": task,
        "metric": metric,
        "score_avg": score_avg,
        "examples": len(records),
        metric: score_avg,
        "old_metric": metrics_old.get("metric"),
        "old_score_avg": metrics_old.get("score_avg"),
    }
    (task_dir / "metrics_official.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if inplace:
        rank_path = task_dir / "rank-0.jsonl"
        if rank_path.is_file():
            _write_jsonl(rank_path, records)
        _write_jsonl(merged_path, records)
        metrics_old["metric"] = metric
        metrics_old["qa_f1"] = score_avg
        metrics_old[metric] = score_avg
        metrics_old["score_avg"] = score_avg
        metrics_path.write_text(
            json.dumps(metrics_old, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    tasks = args.tasks or list(LONGBENCH_ALL_TASKS)
    rescored = 0
    for task in tasks:
        payload = rescore_task(
            output_root=output_root,
            data_root=data_root,
            task=task,
            inplace=args.inplace,
        )
        if payload is None:
            print(json.dumps({"status": "skip_incomplete", "task": task}))
            continue
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        rescored += 1
    print(json.dumps({"rescored": rescored}))


if __name__ == "__main__":
    main()
