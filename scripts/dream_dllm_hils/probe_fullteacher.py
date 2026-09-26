#!/usr/bin/env python3
"""Five teacher rankings share one full dense forward per frozen diagnostic case."""
import argparse
import json
import time
from pathlib import Path

import torch

from dream_dllm_hils.full_dense_teacher import RULES, chunk_scores, dense_pass, token_scores, sampled_candidates
from dream_dllm_hils.diagnostic_helpers import physical_positions
from dream_dllm_hils.lse_calibration import top_indices, group_priority
from dream_dllm_hils.lse_oracle import evidence_recall, retained_mass
from dream_dllm_hils.longbench_eval import append_jsonl_fsync
from dream_dllm_hils.train_fulltext import parse_args, _build_model_and_tokenizer, _unwrap_training_model
from scripts.dream_dllm_hils.diagnose_lse_calibration import restore_model
from scripts.dream_dllm_hils.diagnose_retrieval import make_layout, sha256


@torch.no_grad()
def run(args):
    torch.manual_seed(7)
    training = parse_args(["--config", args.config, "--no_gradient_checkpointing"])
    assert training.model_rope_scaling["factor"] == 16
    assert training.model_rope_scaling["original_max_position_embeddings"] == 2048
    model, tokenizer, _ = _build_model_and_tokenizer(training, torch.device("cuda"))
    restoration = restore_model(model, training, args.checkpoint)
    core = _unwrap_training_model(model)
    rows = [r for r in map(json.loads, Path(args.manifest).read_text().splitlines()) if r["physical_length"] == 32768]
    assert len(rows) == 14
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.rank::args.world_size]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(version="fullteacher-five-rules-v1", restoration=restoration,
                    config_sha256=sha256(args.config), manifest_sha256=sha256(args.manifest),
                    yarn=training.model_rope_scaling, rules=RULES,
                    selection="per-head remote conditional softmax, KV group max, top16; also original-gate sensitivity",
                    normalization="T2-T5 are retrieval preferences, not exact chunk mass",
                    teacher="same current main-LoRA weights, all 28 dense layers; one pass for five rules",
                    phase="initial MASK prefill; no generation")
    previous = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if any(r["identity"] != json.loads(json.dumps(identity)) for r in previous):
        raise ValueError("probe output provenance changed")
    done = {r["case_id"] for r in previous}
    for row in rows:
        if row["case_id"] in done:
            continue
        case, layout = make_layout(row, tokenizer, training, 0)
        positions = layout.predictor_positions[torch.linspace(0, len(layout.predictor_positions)-1, 16).long()][None].cuda()
        batch = dict(input_ids=layout.input_ids[None].cuda(), attention_mask=layout.attention_mask[None].cuda(),
                     position_ids=layout.position_ids[None].cuda())
        values = torch.unique(physical_positions(torch.tensor(sum(case.evidence_values, [])), 64) // 64).cuda()
        facts = torch.unique(physical_positions(torch.tensor(sum(case.evidence_facts, [])), 64) // 64).cuda()
        records, visited = [], []

        def observe(index, q, k, valid, segments):
            visited.append(index)
            if index not in (3, 7, 11, 15, 19, 23, 27):
                return
            raw = token_scores(q, k, positions, valid, segments, 64)
            eligible, local = sampled_candidates(positions, valid, segments, 64, training.local_window)
            exact = chunk_scores(raw, "lse")[0]
            local_lse = torch.logsumexp(exact.masked_fill(~local[0, :, None, None], -torch.inf), -1)
            dropped = ~eligible[0]
            scores_by_rule = {}
            for rule in RULES:
                scores = chunk_scores(raw, rule)[0]
                remote_scores = scores.masked_fill(dropped[:, None, None], -torch.inf)
                head_p = remote_scores.log_softmax(-1)
                indices = top_indices(head_p.max(2).values, 16)
                per_head = top_indices(remote_scores, 16).flatten(1, 2)
                original_gate = top_indices(group_priority(scores, local_lse, dropped), 16)
                scores_by_rule[rule] = dict(value=evidence_recall(indices, dropped, values),
                    fact=evidence_recall(indices, dropped, facts),
                    per_head_value=evidence_recall(per_head, dropped, values),
                    original_gate_sensitivity=evidence_recall(original_gate, dropped, values),
                    retained_true_remote_mass=retained_mass(exact, dropped, indices))
            records.append(dict(layer=index, query_positions=positions[0].tolist(), rules=scores_by_rule))

        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = dense_pass(core, batch, observe)
            gold = tokenizer(" " + " ".join(case.answers), add_special_tokens=False).input_ids
            logits = core.lm_head(hidden[0, layout.predictor_positions[:len(gold)].cuda()])
            nll = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(gold, device="cuda"))
        assert visited == list(range(28)) and len(records) == 7
        result = dict(identity=identity, case_id=row["case_id"], task=row["task"], depth=row["depth"],
                      observations=records, seconds=time.monotonic()-start, gold_nll=float(nll), layers=visited)
        append_jsonl_fsync(output, result)
        print(json.dumps(dict(case_id=row["case_id"], seconds=result["seconds"],
                              recall={rule: sum(x["rules"][rule]["value"]["recall"] for x in records)/7 for rule in RULES})), flush=True)
        del hidden, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "checkpoint", "manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    run(parser.parse_args())
