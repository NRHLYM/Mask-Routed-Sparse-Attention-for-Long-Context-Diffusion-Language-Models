"""CPU helpers for LMK vs token-LSE routing diagnosis."""
from __future__ import annotations

import math

import torch


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if int(x.numel()) < 3:
        return float("nan")
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = float(x.norm() * y.norm())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def rank_desc(scores: torch.Tensor) -> torch.Tensor:
    """1-based rank among finite values; higher score is rank 1."""
    finite = torch.isfinite(scores)
    ranks = torch.full(scores.shape, float("nan"), dtype=torch.float32, device=scores.device)
    if not bool(finite.any()):
        return ranks
    order = torch.argsort(scores.masked_fill(~finite, -1e30), descending=True)
    place = torch.empty_like(order)
    place[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    ranks = place.float()
    return ranks.masked_fill(~finite, float("nan"))


def item_stats(
    *,
    a_lmk: torch.Tensor,
    z_token: torch.Tensor,
    z_evidence: torch.Tensor,
    weight: torch.Tensor,
    correct: torch.Tensor,
    selected: torch.Tensor,
    a_lmk_all: torch.Tensor | None = None,
    correct_all: torch.Tensor | None = None,
) -> dict:
    """Selected-chunk tensors [P, H, K]; optional all-chunk LMK [P, H, C]."""
    valid = selected & torch.isfinite(a_lmk) & torch.isfinite(z_token)
    pair_a, pair_z, pair_w = a_lmk[valid], z_token[valid], weight[valid]
    best_lmk = torch.where(correct & valid, a_lmk, torch.full_like(a_lmk, -1e30))
    best_tok = torch.where(correct & valid, z_token, torch.full_like(z_token, -1e30))
    best_ev = torch.where(correct & valid, z_evidence, torch.full_like(z_evidence, -1e30))
    best_w = torch.where(correct & valid, weight, torch.zeros_like(weight))
    in_sel = (correct & valid).any(dim=-1)
    lmk_rank: list[float] = []
    tok_rank: list[float] = []
    ev_rank: list[float] = []
    fusion_rank: list[float] = []
    lmk_all_rank: list[float] = []
    for p in range(a_lmk.shape[0]):
        for h in range(a_lmk.shape[1]):
            if bool(in_sel[p, h]):
                mask = valid[p, h]
                target = correct[p, h][mask]
                ra = rank_desc(a_lmk[p, h][mask])
                rz = rank_desc(z_token[p, h][mask])
                re = rank_desc(z_evidence[p, h][mask])
                rw = rank_desc(weight[p, h][mask])
                if bool(target.any()):
                    lmk_rank.append(float(ra[target].min()))
                    tok_rank.append(float(rz[target].min()))
                    fusion_rank.append(float(rw[target].min()))
                    if bool(torch.isfinite(re[target]).any()):
                        ev_rank.append(float(re[target][torch.isfinite(re[target])].min()))
            if a_lmk_all is not None and correct_all is not None:
                target_all = correct_all[p, h]
                scores_all = a_lmk_all[p, h]
                if bool(target_all.any()) and bool(torch.isfinite(scores_all[target_all]).any()):
                    ranked = rank_desc(scores_all)
                    lmk_all_rank.append(float(ranked[target_all][torch.isfinite(ranked[target_all])].min()))

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else float("nan")

    return {
        "n_selected": int(valid.sum()),
        "n_correct_selected": int((correct & valid).sum()),
        "support_frac": float(in_sel.float().mean()),
        "mean_lmk_rank": _mean(lmk_rank),
        "rank_lmk_selected": _mean(lmk_rank),
        "rank_fusion_selected": _mean(fusion_rank),
        "rank_lmk_all": _mean(lmk_all_rank),
        "mean_token_rank": _mean(tok_rank),
        "mean_evidence_rank": _mean(ev_rank),
        "corr_lmk_token": pearson(pair_a, pair_z),
        "corr_fusion_lmk": pearson(pair_w, pair_a),
        "corr_fusion_token": pearson(pair_w, pair_z),
        "mean_correct_lmk": float(best_lmk.amax(dim=-1)[in_sel].mean()) if bool(in_sel.any()) else float("nan"),
        "mean_correct_token_lse": float(best_tok.amax(dim=-1)[in_sel].mean()) if bool(in_sel.any()) else float("nan"),
        "mean_correct_evidence_lse": float(best_ev.amax(dim=-1)[in_sel].mean()) if bool(in_sel.any()) else float("nan"),
        "mean_correct_fusion": float(best_w.sum(dim=-1).mean()),
        "mean_fusion_total": float(weight.masked_fill(~valid, 0).sum(dim=-1).mean()),
        "n_rank_obs": len(lmk_rank),
        "n_lmk_all_obs": len(lmk_all_rank),
    }


def classify_failure(
    *,
    support_frac: float,
    mean_lmk_rank: float,
    mean_token_rank: float,
    corr_lmk_token: float,
    corr_fusion_lmk: float,
    k_selected: float = 32.0,
) -> str:
    """Coarse label for whether token evidence belongs in fusion.

    Rank 1 is best among the selected top-k. The cut at 3 follows the
    existing MK-MQ diagnosis (correct fact typically ~3rd of 6 similar
    chunks): worse than that is not a fusion-only problem.
    """
    del k_selected
    if not math.isfinite(support_frac) or support_frac < 0.5:
        return "routing_miss"
    if not math.isfinite(mean_lmk_rank) or mean_lmk_rank > 3.0:
        return "lmk_ranks_wrong"
    if not math.isfinite(mean_token_rank) or mean_token_rank > 3.0:
        return "token_qk_dead"
    if math.isfinite(corr_fusion_lmk) and corr_fusion_lmk < 0.4:
        return "fusion_misuses_scores"
    if math.isfinite(corr_lmk_token) and corr_lmk_token < 0.3:
        return "lmk_token_disagree"
    return "aligned"
