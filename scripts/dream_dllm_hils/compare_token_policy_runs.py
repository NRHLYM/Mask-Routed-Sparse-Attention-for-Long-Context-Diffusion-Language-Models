#!/usr/bin/env python3
"""Build the final training-stability and MFEN comparison for token policies."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_training_stability import summarize_log


RUNS = {
    "dense": {
        "log": Path("outputs/dream-dense-dolma3-2k.log"),
        "metrics": Path(
            "outputs/dream-dense-dolma3-2k/"
            "longbench_mfen_exact/metrics.json"
        ),
    },
    "hils_full_chunks": {
        "log": Path("outputs/dream-hils-dense21-dolma3-2k.log"),
        "metrics": Path(
            "outputs/dream-hils-dense21-dolma3-2k/"
            "longbench_mfen_exact/metrics.json"
        ),
    },
    "hils_global_qk_256": {
        "log": Path(
            "outputs/dream-hils-hisa256-dense21-dolma3-2k.queue.log"
        ),
        "metrics": Path(
            "outputs/dream-hils-hisa256-dense21-dolma3-2k/"
            "longbench_mfen_exact/metrics.json"
        ),
    },
    "hils_entropy_adaptive_256": {
        "log": Path(
            "outputs/dream-hils-entropy256-dense21-dolma3-2k.queue.log"
        ),
        "metrics": Path(
            "outputs/dream-hils-entropy256-dense21-dolma3-2k/"
            "longbench_mfen_exact/metrics.json"
        ),
    },
}


def main() -> None:
    report: dict[str, object] = {}
    for name, paths in RUNS.items():
        missing = [
            str(path) for path in paths.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{name} is incomplete; missing {missing}"
            )
        stability = summarize_log(paths["log"])
        quality = json.loads(
            paths["metrics"].read_text(encoding="utf-8")
        )
        report[name] = {
            "training": stability,
            "mfen": {
                key: quality[key]
                for key in (
                    "qa_f1",
                    "mean_seconds",
                    "p50_seconds",
                    "p95_seconds",
                    "peak_memory_bytes_max",
                    "fallback_max",
                    "examples",
                    "model_variant",
                )
            },
        }

    dense_f1 = report["dense"]["mfen"]["qa_f1"]
    dense_validation = report["dense"]["training"]["validation"]
    dense_final_validation = dense_validation[
        max(dense_validation, key=int)
    ]
    for payload in report.values():
        payload["mfen"]["qa_f1_delta_vs_dense"] = (
            payload["mfen"]["qa_f1"] - dense_f1
        )
        validation = payload["training"]["validation"]
        final_validation = validation[max(validation, key=int)]
        payload["training"]["final_validation_delta_vs_dense"] = (
            final_validation - dense_final_validation
        )

    output = Path("outputs/chunk-token-policy-comparison.json")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
