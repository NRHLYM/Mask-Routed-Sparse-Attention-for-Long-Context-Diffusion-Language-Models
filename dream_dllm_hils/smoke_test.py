"""Tiny-model smoke test for the Dream+HiLS prototype."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from dream_dllm_hils.attention import install_dream_sparse_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/ma-user/work/models/Dream-v0-Base-7B")
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg.hidden_size = 32
    cfg.intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_hidden_layers = 4
    cfg.vocab_size = 128
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg.pad_token_id = 0
    cfg.mask_token_id = 3
    cfg.max_position_embeddings = 256
    cfg.rope_theta = 10000.0

    model = AutoModel.from_config(cfg, trust_remote_code=True)
    plan = install_dream_sparse_attention(model, interleave=2, local_window=8, chunk_size=8, topk=2)
    model.train()

    input_ids = torch.randint(4, cfg.vocab_size, (2, 56), dtype=torch.long)
    input_ids[:, 10:14] = cfg.mask_token_id
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand_as(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, 9:13] = torch.randint(4, cfg.vocab_size, (2, 4), dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    loss = F.cross_entropy(outputs.logits.view(-1, cfg.vocab_size), labels.view(-1), ignore_index=-100)
    loss.backward()
    print("smoke ok")
    print("hils layers", plan.hils_layers)
    print("loss", float(loss.detach()))


if __name__ == "__main__":
    main()
