from types import SimpleNamespace

import pytest
import torch

from dream_dllm_hils.fastdllm_v1 import DreamHiLSFastDLLM, PrefillOutput, CachedForwardOutput
from dream_dllm_hils.longbench_eval import build_generation_layout, build_fastdllm_block_layouts
from dream_dllm_hils.rank_audit import evidence_rank_audit


class ToyDecoder(DreamHiLSFastDLLM):
    def prefill(self, input_ids, attention_mask, position_ids, landmark_positions=None):
        logits = torch.zeros(1, input_ids.shape[1], 6)
        # A later answer position is initially more confident than the first.
        logits[0, 0, 1 if input_ids[0, 2] == 0 else 2] = 1 if input_ids[0, 2] == 0 else 12
        logits[0, 1, 3] = 12
        return PrefillOutput(logits, None)

    def cached_forward(self, ids, block, cache, landmark_positions=None):
        output = self.prefill(ids, None, None)
        return CachedForwardOutput(output.logits[:, block.query_positions], block.query_positions, 0)


@pytest.mark.parametrize("cached", [False, True])
def test_confidence_bootstrap_does_not_force_low_confidence_initial_eos(cached):
    layout = build_generation_layout(prompt_ids=[4], answer_tokens=2, physical_length=8,
                                    chunk_size=4, mask_token_id=0, pad_token_id=5, landmark_token_id=0)
    args = dict(input_ids=layout.input_ids[None], attention_mask=layout.attention_mask[None],
                position_ids=layout.position_ids[None], landmark_positions=layout.landmark_positions,
                blocks=build_fastdllm_block_layouts(layout, 2, 4))
    old, old_stats = ToyDecoder(model=None, mask_token_id=0, use_cache=cached).generate(**args)
    new, new_stats = ToyDecoder(model=None, mask_token_id=0, use_cache=cached, bootstrap="confidence").generate(**args)
    assert old[0, layout.answer_positions].tolist() == [1, 3]
    assert new[0, layout.answer_positions].tolist() == [2, 3]
    assert torch.equal(old[0, layout.landmark_positions], new[0, layout.landmark_positions])
    assert new_stats.full_prefills + new_stats.cached_forwards == 2
    assert old_stats.full_prefills + old_stats.cached_forwards == 2


def test_invalid_bootstrap_rejected():
    with pytest.raises(ValueError, match="bootstrap"):
        ToyDecoder(model=None, mask_token_id=0, bootstrap="suppress_eos")


def test_confident_eos_is_not_suppressed_and_can_finish_in_prefill():
    class EosDecoder(ToyDecoder):
        def prefill(self, input_ids, attention_mask, position_ids, landmark_positions=None):
            logits = torch.zeros(1, input_ids.shape[1], 6)
            logits[..., 1] = 12
            return PrefillOutput(logits, None)
    layout = build_generation_layout(prompt_ids=[4], answer_tokens=2, physical_length=8,
                                    chunk_size=4, mask_token_id=0, pad_token_id=5, landmark_token_id=0)
    generated, stats = EosDecoder(model=None, mask_token_id=0, bootstrap="confidence").generate(
        input_ids=layout.input_ids[None], attention_mask=layout.attention_mask[None], position_ids=layout.position_ids[None],
        blocks=build_fastdllm_block_layouts(layout, 2, 4))
    assert generated[0, layout.answer_positions].tolist() == [1, 1]
    assert stats.full_prefills == 1 and stats.cached_forwards == 0


def test_rank_audit_finds_high_scoring_missed_token_and_not_local_or_invalid():
    scores = torch.tensor([[[10., 4., 3., float("-inf"), 9., 5., 2., float("-inf")]]])
    local = torch.zeros_like(scores, dtype=torch.bool)
    local[..., 0] = True
    before = torch.zeros_like(local)
    before[..., 1:3] = True
    priority = torch.tensor([[[2., 1.]]])
    out = evidence_rank_audit(scores, before, before, local, [torch.tensor([0, 4, 7])], priority, budget=2, chunk_size=4)
    assert out["remote_evidence_token_units"] == 1
    assert out["missed_strict_full_top_budget_units"] == 1
    assert out["examples"][0]["chunk_rank_best"] == 2
    assert out["examples"][0]["token_position"] == 4


def test_rank_audit_does_not_count_ties_as_strict_high_scores():
    scores = torch.ones(1, 1, 8)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    before = mask.clone()
    before[..., :2] = True
    out = evidence_rank_audit(scores, before, before, mask, [torch.tensor([4])], None, budget=2)
    assert out["missed_evidence_token_units"] == 1
    assert out["missed_strict_full_top_budget_units"] == 0


def test_paired_training_template_keeps_code_and_depth_across_lengths():
    from scripts.dream_dllm_hils.bridge_generation import PairedSynthesizer, find_span
    class Tokenizer:
        eos_token_id = 999
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[ord(c) for c in text])
    values = []
    for length in (512, 2048):
        synth = PairedSynthesizer(Tokenizer(), 77, 0.5)
        clean, target = synth.synthesize(torch.full((length,), 1000), task_id=0)
        answer = clean[target][:-1].tolist()
        fact = [int(clean[p]) for p in synth.fact]
        span = find_span(fact, answer)
        assert len(span) == 8 and len(clean) == length
        values.append(answer)
    assert values[0] == values[1]


def gate_fixture():
    import hashlib
    import json
    from dream_dllm_hils.bridge_gate import MODES
    manifest = []
    for length in (2048, 8192, 16384, 32768):
        for i in range(12):
            manifest.append(dict(case_id=f"{length}-{i}", kind="paired", pair_id=i, physical_length=length,
                                 answer_tokens=32, answers=["12345678"], prompt_ids=[length, i]))
    for i in range(12):
        manifest.append(dict(case_id=f"lb-{i}", kind="longbench", physical_length=32768,
                             answer_tokens=32, answers=["answer"], prompt_ids=[i]))
    bridge, controls = [], []
    for r in manifest:
        for model in ("dsa", "hils"):
            base = dict(r, model=model, manifest_sha256="manifest", checkpoint_sha256=model, config_sha256=model,
                        prompt_sha256=hashlib.sha256(json.dumps(r["prompt_ids"]).encode()).hexdigest(),
                        fallback_count=0, raw_answer_ids=[2] * 32, score=0.5, code_exact_match=1.,
                        full_prefills=1, cached_forwards=1, trace=[{}, {}])
            if r["kind"] != "paired" or (r["pair_id"] < 2 and r["physical_length"] in (2048, 32768)):
                bridge.extend(dict(base, mode=mode, cached_forwards=int(mode.endswith("_cached"))) for mode in MODES)
            if r["kind"] == "paired":
                controls.extend(dict(base, mode=mode) for mode in ("first_token_cached", "confidence_cached"))
    return manifest, bridge, controls


def test_gate_requires_coverage_and_positive_controls():
    from dream_dllm_hils.bridge_gate import assess_controls
    manifest, bridge, controls = gate_fixture()
    out = assess_controls(manifest, bridge, controls, manifest_sha256="manifest")
    assert out["eligible_lengths"] == [2048, 8192, 16384, 32768]
    for r in controls:
        r["code_exact_match"] = 0.
    assert assess_controls(manifest, bridge, controls, manifest_sha256="manifest")["eligible_lengths"] == []
    with pytest.raises(ValueError, match="coverage"):
        assess_controls(manifest, bridge, controls[:-1], manifest_sha256="manifest")


@pytest.mark.parametrize("field,value", [("prompt_sha256", "changed"), ("checkpoint_sha256", "changed"),
                                       ("fallback_count", 1), ("manifest_sha256", "wrong")])
def test_gate_rejects_unpaired_or_invalid_results(field, value):
    from dream_dllm_hils.bridge_gate import assess_controls
    manifest, bridge, controls = gate_fixture()
    controls[0][field] = value
    with pytest.raises(ValueError):
        assess_controls(manifest, bridge, controls, manifest_sha256="manifest")


def test_gate_stops_oracles_on_longbench_regression():
    from dream_dllm_hils.bridge_gate import assess_controls
    manifest, bridge, controls = gate_fixture()
    for r in bridge:
        if r["mode"] == "confidence_cached" and r["kind"] == "longbench":
            r["score"] = 0.
    out = assess_controls(manifest, bridge, controls, manifest_sha256="manifest")
    assert not out["bridge_passed"] and not out["eligible_lengths"]
