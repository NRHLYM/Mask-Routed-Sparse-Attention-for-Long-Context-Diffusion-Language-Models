import pytest
import torch
from types import SimpleNamespace

from dream_dllm_hils.diagnostic_helpers import (
    corrupt_background, coverage_metrics, dense_reference, force_chunks,
    force_tokens, make_case, observer_error_limit, physical_positions, score_codes, support_mask,
)
from dream_dllm_hils.longbench_eval import build_generation_layout
from dream_dllm_hils.diagnostic_probe import summarize_query_route_collapse


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(x) for x in text])


def test_route_collapse_summary_detects_identical_queries_and_support():
    queries = torch.ones(4, 2, 3)
    indices = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
        ]
    )
    result = summarize_query_route_collapse(queries, indices)
    assert result["route_query_same_head_cosine_mean"] == pytest.approx(1)
    assert result["route_query_flat_cosine_mean"] == pytest.approx(1)
    assert result["route_topk_between_query_jaccard_mean"] == pytest.approx(1)
    assert result["route_topk_between_query_exact_fraction"] == 1
    assert result["route_unique_chunks_per_kv_head_mean"] == 2


def test_route_collapse_summary_counts_diverse_support():
    queries = torch.eye(4).reshape(4, 1, 4)
    indices = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]])
    result = summarize_query_route_collapse(queries, indices)
    assert result["route_query_same_head_cosine_mean"] == pytest.approx(0)
    assert result["route_topk_between_query_jaccard_mean"] == pytest.approx(0)
    assert result["route_topk_between_query_exact_fraction"] == 0
    assert result["route_unique_chunks_per_kv_head_mean"] == 8


def test_route_collapse_summary_conditions_wrong_overlap_on_evidence_miss():
    queries = torch.eye(4).reshape(4, 1, 4)
    indices = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            [[9, 5], [6, 7]],
            [[8, 9], [10, 11]],
        ]
    )
    result = summarize_query_route_collapse(
        queries, indices, evidence_chunks=torch.tensor([9])
    )
    assert result["route_evidence_query_kv_hit_rate"] == pytest.approx(0.25)
    assert result["route_evidence_query_chunk_recall"] == pytest.approx(0.5)
    assert result["route_evidence_query_any_chunk_hit_rate"] == pytest.approx(0.5)
    assert result["route_evidence_union_chunk_recall"] == pytest.approx(1)
    assert result["route_evidence_union_all_chunks_hit"] == 1
    assert result["route_evidence_all_queries_miss"] == 0
    assert result["route_evidence_missed_query_pair_count"] == 1
    assert result["route_evidence_missed_query_pair_wrong_jaccard_mean"] == pytest.approx(1)


def test_route_collapse_summary_reports_collective_evidence_miss():
    queries = torch.eye(3).reshape(3, 1, 3)
    indices = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]]])
    result = summarize_query_route_collapse(
        queries, indices, evidence_chunks=torch.tensor([9])
    )
    assert result["route_evidence_query_any_chunk_hit_rate"] == 0
    assert result["route_evidence_union_chunk_recall"] == 0
    assert result["route_evidence_all_queries_miss"] == 1
    assert result["route_evidence_missed_query_pair_count"] == 3
    assert result["route_evidence_missed_query_pair_wrong_jaccard_mean"] == 0


@pytest.mark.parametrize("task", ["single", "multi", "chain"])
@pytest.mark.parametrize("depth", [0.1, 0.5, 0.9])
def test_manifest_spans_and_masking(task, depth):
    tok = CharacterTokenizer()
    case = make_case(tok, [120] * 3000, 1984, task=task, seed=9202609, depth=depth)
    assert len(case.prompt_ids) == 1984
    assert len(case.evidence_values) == (1 if task == "single" else 2)
    layout = build_generation_layout(prompt_ids=case.prompt_ids, answer_tokens=32,
        physical_length=2048, chunk_size=64, mask_token_id=999, pad_token_id=0, landmark_token_id=998)
    for span in case.evidence_values:
        logical = torch.tensor(span)
        physical = physical_positions(logical, 64)
        assert torch.equal(layout.input_ids[physical], torch.tensor(case.prompt_ids)[logical])
        assert (physical % 64 != 63).all()
    low = corrupt_background(case, 0.5, 999)
    high = corrupt_background(case, 0.9, 999)
    for i, token in enumerate(low):
        if token == 999:
            assert high[i] == 999
    for span in case.evidence_facts:
        for i in span:
            assert high[i] == case.prompt_ids[i]
    assert high[case.question_start:] == case.prompt_ids[case.question_start:]
    assert case == make_case(tok, [120] * 3000, 1984, task=task, seed=9202609, depth=depth)


def test_oracle_chunks_replace_not_append_and_exclude_local():
    indices = torch.tensor([[[0, 2, 4], [0, 1, 3]], [[1, 2, -1], [0, 3, -1]]], dtype=torch.int32)
    priority = torch.arange(6).float().expand(2, 2, 6)
    dropped = torch.zeros(2, 6, dtype=torch.bool)
    dropped[1, 5] = True
    out = force_chunks(indices, priority, [3, 5], dropped)
    assert torch.equal((out >= 0).sum(-1), (indices >= 0).sum(-1))
    assert out[0].tolist() == [[3, 4, 5], [1, 3, 5]]
    assert 5 not in out[1].tolist()[0]
    assert all(3 in row for row in out[1].tolist())
    assert torch.equal(indices[0, 0], torch.tensor([0, 2, 4]))


def test_oracle_token_budget_and_missing_chunk():
    keep = torch.tensor([[[[1, 1, 0, 0], [0, 1, 0, 0]]]], dtype=torch.uint8)
    scores = torch.tensor([[[[0., 1., 2., -torch.inf], [3., 4., 5., -torch.inf]]]])
    positions = torch.tensor([[[[0, 1, 2, 3], [8, 9, 10, 11]]]])
    fixed = force_tokens(keep, scores, positions, torch.tensor([2, 10, 100]))
    assert fixed.sum() == keep.sum() == 3
    assert fixed[0, 0, 0, 2] == fixed[0, 0, 1, 2] == 1
    assert fixed[..., -1].sum() == 0
    assert torch.equal(force_tokens(keep, scores, positions, torch.tensor([100])), keep)


def test_invalid_slots_cannot_leak_into_support():
    ids = torch.tensor([[[0, -1, 1]]])
    valid = torch.ones(8, dtype=torch.bool)
    valid[5] = False
    mask = support_mask(ids, valid, 4)
    assert mask[0, 0].tolist() == [True, True, True, False, True, False, True, False]
    keep = torch.zeros(1, 1, 3, 4, dtype=torch.uint8)
    keep[..., 1, :] = 1
    assert not support_mask(ids, valid, 4, keep).any()


def test_mass_and_conditional_recall_are_separate():
    q = torch.zeros(1, 2, 4)
    k = torch.zeros(8, 1, 4)
    valid = torch.ones(8, dtype=torch.bool)
    _, probs = dense_reference(q, k, valid)
    before = torch.zeros(1, 1, 8, dtype=torch.bool)
    before[..., 4:8] = True
    after = before.clone()
    after[..., 6:] = False
    local = torch.zeros_like(before)
    local[..., :2] = True
    metrics = coverage_metrics(probs, before, after, local, [torch.tensor([4]), torch.tensor([6])])
    assert metrics["candidate_mass"] == 0.75
    assert metrics["retained_mass"] == 0.5
    assert metrics["conditional_retention"] == 0.5
    assert metrics["remote_evidence_recall_before"] == 1
    assert metrics["remote_evidence_recall_after"] == 0.5
    assert metrics["all_evidence_before"] == 1
    assert metrics["all_evidence_after"] == 0


def test_scoring_penalizes_extra_or_wrong_order_codes():
    assert score_codes("12345678 87654321", ["12345678", "87654321"])["code_exact_match"] == 1
    assert score_codes("87654321 12345678", ["12345678", "87654321"])["code_exact_match"] == 0
    assert score_codes("12345678 88888888", ["12345678"])["code_precision"] == 0.5
    assert score_codes("123456789", ["12345678"])["code_recall"] == 0


def test_oracle_rejects_duplicate_native_chunks():
    with pytest.raises(ValueError, match="duplicates"):
        force_chunks(torch.tensor([[[0, 0]]]), torch.zeros(1, 1, 3), [2], torch.zeros(1, 3).bool())


def fake_probe(variant):
    from dream_dllm_hils.diagnostic_probe import RoutingProbe
    probe = RoutingProbe(SimpleNamespace())
    probe.variant, probe.observe = variant, False
    probe.positions = torch.tensor([190, 191])
    probe.active = torch.tensor([190])
    probe.module = SimpleNamespace(local_window=0)
    probe.facts = [torch.tensor([0, 1])]
    probe.interventions = {"changed_chunk_slots": 0, "changed_token_slots": 0}
    return probe


def test_probe_chunk_oracle_recomputes_raw_scores_only_on_active_rows():
    probe = fake_probe("oracle_chunk")
    torch.manual_seed(9)
    q, landmarks = torch.randn(1, 2, 7, 4), torch.randn(1, 4, 1, 7, 4)
    lse, bias = torch.randn(1, 2, 1, 7), torch.randn(1, 4, 1, 7) * 10
    indices = torch.tensor([[[[1, 2]], [[1, 2]]]], dtype=torch.int32)
    scores = torch.full((1, 2, 7, 2), 999.)
    original = lambda *a, **kw: (indices, scores, None)
    out = probe._route(original, (q, landmarks, lse, bias, torch.zeros(1, 2, 4).bool()), {})
    assert out[2] is None and 0 in indices[0, 0, 0]
    assert indices[0, 1, 0].tolist() == [1, 2] and (scores[0, 1] == 999).all()
    raw = torch.einsum("gd,cgd->gc", q[0, 0], landmarks[0, :, 0]) / 2
    assert torch.allclose(scores[0, 0], raw[:, indices[0, 0, 0].long()])
    assert (indices >= 0).sum() == 4


def test_probe_token_oracle_preserves_native_inactive_rows_and_budget():
    probe = fake_probe("oracle_token")
    q, k = torch.zeros(1, 2, 7, 4), torch.zeros(1, 192, 1, 4)
    indices = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.int32)
    keep = torch.zeros(1, 2, 1, 2, 64, dtype=torch.int8)
    keep[..., 0, 4:7] = 1
    old = keep.clone()
    out = probe._tokens(lambda *a, **kw: keep, (q, k, indices, torch.ones(1, 192).bool(), 64), {})
    assert torch.equal(out.sum((-1, -2)), old.sum((-1, -2)))
    assert torch.equal(out[:, 1], old[:, 1]) and out[0, 0, 0, 0, :2].all()


def test_flat_gate_is_exact_selected_softmax_and_excludes_landmarks():
    probe = fake_probe("flat_gate")
    q, k = torch.zeros(1, 2, 7, 4), torch.zeros(1, 192, 1, 4)
    v = torch.zeros_like(k)
    indices = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.int32)
    keep = torch.zeros(1, 2, 1, 2, 64, dtype=torch.int8)
    keep[..., 0, :2], keep[..., 1, :4] = 1, 1
    weights = torch.full((1, 2, 7, 2), 0.1)
    captured = {}
    def original(q, k, v, weight, *args, **kwargs):
        captured["weight"] = weight
        return torch.zeros_like(q)
    probe._hils_selected(original, (q, k, v, weights, indices, torch.ones(1, 192).bool(), 64), {"token_keep": keep})
    # Two and four remote tokens plus 63 local text keys, no landmark keys.
    expected = torch.tensor([2 / 69, 4 / 69]).expand(7, -1)
    assert torch.allclose(captured["weight"][0, 0], expected)
    assert torch.equal(captured["weight"][0, 1], weights[0, 1])
    rows, local_gate = probe.gate_update
    assert rows.tolist() == [0] and torch.allclose(local_gate, torch.full_like(local_gate, 63 / 69))
    local = probe._local(probe.positions, 192, 1, torch.ones(192).bool())
    assert local.sum(-1).tolist() == [[63], [63]] and not local[..., 63::64].any()


def test_probe_dense_counterfactual_has_no_landmark_probability():
    probe = fake_probe("baseline")
    probe.observe, probe.step = True, 0
    probe.evidence, probe.layer, probe.metadata, probe.route_summary = [torch.tensor([0])], 3, {}, {}
    q, k = torch.zeros(1, 2, 7, 4), torch.zeros(1, 192, 1, 4)
    v = torch.zeros_like(k)
    v[:, 63::64] = 100
    indices = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.int32)
    output = torch.randn_like(q)
    args = (q, k, v, torch.zeros(1, 2, 7, 2), indices, torch.ones(1, 192).bool(), 64)
    before = [x.clone() for x in args[:-1]]
    result = probe._hils_selected(lambda *a, **kw: output, args, {})
    assert result is output
    assert all(torch.equal(a, b) for a, b in zip(args[:-1], before))
    record, _, teacher = probe.pending
    assert not teacher.any() and record["actual_total_tokens_mean"] == 189
    assert record["candidate_mass"] == pytest.approx(1.0)


def test_observer_gate_uses_native_repeats_but_keeps_absolute_caps():
    assert observer_error_limit([0, 0, 0]) == 1e-6
    assert observer_error_limit([0.009, 0.01, 0.011]) == pytest.approx(0.022)
    assert observer_error_limit([0.019]) == 0.03
    with pytest.raises(AssertionError, match="too unstable"):
        observer_error_limit([0.0201])
    for values in ([], [float("nan")], [float("inf")], [-0.1]):
        with pytest.raises(ValueError):
            observer_error_limit(values)


@pytest.mark.parametrize("failure", [None, "duplicate", "missing", "fallback", "provenance", "layer", "cache"])
def test_summary_coverage_gates(failure):
    import copy
    from scripts.summarize_retrieval_diagnostics import validate
    record = {"identity": {"model": "hils", "phase": "generate", "manifest_sha256": "test-only"},
              "case_id": "test-fixture", "variant": "baseline", "background_mask_ratio": 0,
              "fallback_count": 0, "full_prefills": 1, "cached_forwards": 3,
              "observations": [{"layer": layer, "forward_index": 0} for layer in (3, 7, 11, 15, 19, 23, 27)]}
    records = [copy.deepcopy(record)]
    if failure == "duplicate": records.append(copy.deepcopy(record))
    elif failure == "missing": records.clear()
    elif failure == "fallback": records[0]["fallback_count"] = 1
    elif failure == "provenance": records[0]["identity"]["manifest_sha256"] = "wrong"
    elif failure == "layer": records[0]["observations"].pop()
    elif failure == "cache": records[0]["cached_forwards"] = 0
    expected = {("hils", "test-fixture", "baseline", 0)}
    if failure:
        with pytest.raises(AssertionError):
            validate(records, expected, phase="generate", manifest_sha="test-only")
    else:
        validate(records, expected, phase="generate", manifest_sha="test-only")
