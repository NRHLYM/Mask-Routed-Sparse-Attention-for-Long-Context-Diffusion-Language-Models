from dream_dllm_hils.lmk_token_lse_stats import classify_failure, item_stats, pearson, rank_desc
import torch


def test_rank_desc_prefers_higher_scores():
    ranks = rank_desc(torch.tensor([0.1, 0.9, float("nan"), 0.4]))
    assert float(ranks[1]) == 1
    assert float(ranks[3]) == 2
    assert float(ranks[0]) == 3
    assert not torch.isfinite(ranks[2])


def test_pearson_perfect_and_empty():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(pearson(x, x) - 1.0) < 1e-5
    assert abs(pearson(x, -x) + 1.0) < 1e-5
    assert pearson(x[:1], x[:1]) != pearson(x[:1], x[:1])  # nan


def test_item_stats_correct_chunk_ranks():
    a = torch.tensor([[[0.2, 0.9, 0.1]]])
    z = torch.tensor([[[0.8, 0.1, 0.0]]])
    w = torch.tensor([[[0.1, 0.7, 0.2]]])
    selected = torch.tensor([[[True, True, True]]])
    correct = torch.tensor([[[True, False, False]]])
    stats = item_stats(
        a_lmk=a,
        z_token=z,
        z_evidence=z,
        weight=w,
        correct=correct,
        selected=selected,
    )
    assert stats["mean_lmk_rank"] == 2
    assert stats["mean_token_rank"] == 1
    assert stats["rank_fusion_selected"] == 3
    assert stats["support_frac"] == 1.0
    assert stats["corr_fusion_lmk"] > 0


def test_rank_lmk_all_uses_unselected_chunks():
    a = torch.tensor([[[0.2, 0.1]]])
    z = torch.tensor([[[0.2, 0.1]]])
    w = torch.tensor([[[0.9, 0.1]]])
    selected = torch.ones(1, 1, 2, dtype=torch.bool)
    correct_sel = torch.tensor([[[False, True]]])
    a_all = torch.tensor([[[0.05, 0.9, 0.4]]])
    correct_all = torch.tensor([[[False, True, False]]])
    stats = item_stats(
        a_lmk=a,
        z_token=z,
        z_evidence=z,
        weight=w,
        correct=correct_sel,
        selected=selected,
        a_lmk_all=a_all,
        correct_all=correct_all,
    )
    assert stats["rank_lmk_all"] == 1
    assert stats["rank_fusion_selected"] == 2


def test_classify_maps_the_three_hypotheses():
    assert classify_failure(
        support_frac=0.2,
        mean_lmk_rank=1.2,
        mean_token_rank=1.1,
        corr_lmk_token=0.9,
        corr_fusion_lmk=0.9,
    ) == "routing_miss"
    assert classify_failure(
        support_frac=0.95,
        mean_lmk_rank=8.0,
        mean_token_rank=1.5,
        corr_lmk_token=0.2,
        corr_fusion_lmk=0.9,
    ) == "lmk_ranks_wrong"
    assert classify_failure(
        support_frac=0.95,
        mean_lmk_rank=1.4,
        mean_token_rank=9.0,
        corr_lmk_token=0.1,
        corr_fusion_lmk=0.8,
    ) == "token_qk_dead"
    assert classify_failure(
        support_frac=0.95,
        mean_lmk_rank=1.5,
        mean_token_rank=1.6,
        corr_lmk_token=0.8,
        corr_fusion_lmk=0.1,
    ) == "fusion_misuses_scores"
