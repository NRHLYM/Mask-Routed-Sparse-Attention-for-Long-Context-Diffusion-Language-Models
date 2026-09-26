"""Predeclared execution, pairing, and positive-control gates for oracle jobs."""
from collections import defaultdict
import hashlib
import json
import math

MODES = ("legacy_exact", "first_token_no_cache", "first_token_cached", "confidence_no_cache", "confidence_cached")


def assess_controls(manifest, bridge, controls, *, manifest_sha256):
    cases = {r["case_id"]: r for r in manifest}
    if len(cases) != len(manifest):
        raise ValueError("duplicate manifest case")
    bridge_ids = {r["case_id"] for r in manifest if r["kind"] != "paired" or (r["pair_id"] < 2 and r["physical_length"] in (2048, 32768))}
    control_ids = {r["case_id"] for r in manifest if r["kind"] == "paired"}
    hashes, checkpoints, configs = defaultdict(set), defaultdict(set), defaultdict(set)
    for data, ids, modes in ((bridge, bridge_ids, MODES), (controls, control_ids, ("first_token_cached", "confidence_cached"))):
        expected = {(model, case, mode) for model in ("hils", "dsa") for case in ids for mode in modes}
        observed = [(r["model"], r["case_id"], r["mode"]) for r in data]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("missing, duplicate, or unexpected control coverage")
        for r in data:
            case = cases[r["case_id"]]
            if r["manifest_sha256"] != manifest_sha256 or r["fallback_count"] != 0:
                raise ValueError("provenance or fallback mismatch")
            if r["answers"] != case["answers"] or r["physical_length"] != case["physical_length"]:
                raise ValueError("control case mismatch")
            expected_prompt = hashlib.sha256(json.dumps(case["prompt_ids"]).encode()).hexdigest()
            if r["prompt_sha256"] != expected_prompt:
                raise ValueError("prompt does not match manifest")
            if len(r["raw_answer_ids"]) != case["answer_tokens"]:
                raise ValueError("incomplete raw output")
            score = r["score"] if case["kind"] == "longbench" else r["code_exact_match"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid score")
            if r["mode"].endswith("_cached"):
                expected_prefills = math.ceil(case["answer_tokens"] / 32)
                if r["full_prefills"] != expected_prefills:
                    raise ValueError("incorrect prefill count")
                if r["cached_forwards"] == 0 and len(r.get("trace", [])) > expected_prefills:
                    raise ValueError("cache was not used")
            elif r["cached_forwards"] != 0:
                raise ValueError("uncached control used cache")
            hashes[r["case_id"]].add(r["prompt_sha256"])
            checkpoints[r["model"]].add(r["checkpoint_sha256"])
            configs[r["model"]].add(r["config_sha256"])
    if any(len(v) != 1 for groups in (hashes, checkpoints, configs) for v in groups.values()):
        raise ValueError("unpaired input, checkpoint, or config")
    anchors, bridge_ok = {}, True
    for model in ("dsa", "hils"):
        means = {}
        for mode in MODES:
            group = [r for r in bridge if r["model"] == model and r["mode"] == mode and r["kind"] == "longbench"]
            if len(group) != 12:
                raise ValueError("need 12 LongBench anchor examples")
            means[mode] = sum(r["score"] for r in group) / len(group)
        passed = all(means["confidence_cached"] >= means[other] - 0.10 for other in ("legacy_exact", "confidence_no_cache", "first_token_cached"))
        if model == "dsa":
            passed &= means["legacy_exact"] > 0 and means["confidence_cached"] > 0
        anchors[model] = dict(mean_qa_f1=means, passed=bool(passed))
        bridge_ok &= passed
    eligibility = []
    for length in (2048, 8192, 16384, 32768):
        group = [r for r in controls if r["model"] == "dsa" and r["mode"] == "confidence_cached" and r["physical_length"] == length]
        if len(group) != 12:
            raise ValueError("need 12 predeclared cases per length")
        correct = sum(r["code_exact_match"] for r in group)
        eligibility.append(dict(length=length, dsa_correct=int(correct), n=12, passed=bool(bridge_ok and correct >= 3)))
    return dict(execution_passed=True, bridge_passed=bool(bridge_ok), anchors=anchors, per_length=eligibility,
                eligible_lengths=[g["length"] for g in eligibility if g["passed"]],
                preregistered_rule="12 cases/length; DSA confidence_cached EM >= 3/12; paired LongBench mean F1 drop <= 0.10 vs each of legacy_exact/confidence_no_cache/first_token_cached, for both models",
                caveat="Pilot guardrails, not statistical noninferiority. Evaluate every case at an eligible length; never select only successful cases.")
