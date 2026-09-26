#!/usr/bin/env python3
"""Summarize loss, validation, clipping, and gradient groups from trainer logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


TRAIN_RE = re.compile(
    r"rank=0 step=(?P<step>\d+) loss=(?P<loss>[0-9.eE+-]+).*?"
    r"grad_norm=(?P<grad>[0-9.eE+-]+).*?"
    r"(?:q_grad_norm=(?P<q>[0-9.eE+-]+).*?"
    r"k_grad_norm=(?P<k>[0-9.eE+-]+).*?"
    r"v_grad_norm=(?P<v>[0-9.eE+-]+).*?"
    r"o_grad_norm=(?P<o>[0-9.eE+-]+).*?)?"
    r"(?:clip_applied=(?P<clipped>[01]).*?)?"
    r"lr=(?P<lr>[0-9.eE+-]+)"
)
VALIDATION_RE = re.compile(
    r"rank=0 step=(?P<step>\d+) "
    r"validation_loss=(?P<loss>[0-9.eE+-]+)"
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]


def summarize_log(path: Path) -> dict[str, object]:
    rows: dict[int, dict[str, float]] = {}
    validation: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            values = train_match.groupdict()
            rows[int(values["step"])] = {
                name: float(value)
                for name, value in values.items()
                if name != "step" and value is not None
            }
        validation_match = VALIDATION_RE.search(line)
        if validation_match:
            validation[int(validation_match.group("step"))] = float(
                validation_match.group("loss")
            )
    if not rows:
        raise ValueError(f"no rank-0 training rows found in {path}")

    ordered_steps = sorted(rows)
    losses = [rows[step]["loss"] for step in ordered_steps]
    gradients = [rows[step]["grad"] for step in ordered_steps]
    quarter = max(1, len(losses) // 4)
    summary: dict[str, object] = {
        "path": str(path),
        "logged_steps": len(rows),
        "first_step": ordered_steps[0],
        "last_step": ordered_steps[-1],
        "loss_mean": statistics.mean(losses),
        "loss_std": statistics.pstdev(losses),
        "loss_first_quarter_mean": statistics.mean(losses[:quarter]),
        "loss_last_quarter_mean": statistics.mean(losses[-quarter:]),
        "grad_median": statistics.median(gradients),
        "grad_p90": percentile(gradients, 0.9),
        "grad_max": max(gradients),
        "validation": {
            str(step): loss for step, loss in sorted(validation.items())
        },
    }
    if validation:
        validation_steps = sorted(validation)
        summary["validation_delta"] = (
            validation[validation_steps[-1]]
            - validation[validation_steps[0]]
        )
    if all("clipped" in row for row in rows.values()):
        summary["clip_fraction"] = statistics.mean(
            row["clipped"] for row in rows.values()
        )
    for name in ("q", "k", "v", "o"):
        available = [
            row[name] for row in rows.values() if name in row
        ]
        if available:
            summary[f"{name}_grad_median"] = statistics.median(available)
            summary[f"{name}_grad_p90"] = percentile(available, 0.9)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        path.stem: summarize_log(path)
        for path in args.logs
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
