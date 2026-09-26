#!/usr/bin/env python3
"""Two-rank CPU integration for loss scaling, no_sync, checkpoint, and resume."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dream_dllm_hils.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from dream_dllm_hils.distributed import (
    all_gather_object,
    global_target_count,
    init_distributed,
    scale_local_loss_for_ddp,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, values):
        return self.weight * values


def _optimizer_scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    context = init_distributed()
    assert context.world_size == 2
    assert context.device.type == "cpu"

    local_ids = [context.rank, context.rank + 2]
    gathered_ids = all_gather_object(local_ids)
    if context.rank == 0:
        assert set(gathered_ids[0]).isdisjoint(gathered_ids[1])

    model = DistributedDataParallel(TinyModel())
    optimizer, scheduler = _optimizer_scheduler(model)
    counts = [torch.tensor(1), torch.tensor(1)]
    global_count = global_target_count(counts, device=context.device)
    optimizer.zero_grad(set_to_none=True)
    for micro, sample_id in enumerate(local_ids):
        sync = model.no_sync() if micro == 0 else torch.enable_grad()
        with sync:
            loss_sum = model(torch.tensor([float(sample_id + 1)])).sum()
            scale_local_loss_for_ddp(
                loss_sum, global_count=global_count, world_size=2
            ).backward()
    optimizer.step()
    scheduler.step()

    weights = [None, None]
    dist.all_gather_object(weights, model.module.weight.detach().clone())
    torch.testing.assert_close(weights[0], weights[1])
    save_training_checkpoint(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.output_dir,
        step=1,
        sampler_epoch=0,
        batches_in_epoch=2,
        config={"kind": "tiny-ddp"},
        dataset_manifest_hash="tiny-dataset",
        rank=context.rank,
        world_size=context.world_size,
        git_commit="tiny-integration",
    )

    restored = TinyModel()
    restored_optimizer, restored_scheduler = _optimizer_scheduler(restored)
    state = load_training_checkpoint(
        checkpoint_dir=args.output_dir / "step-1",
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        rank=context.rank,
        restore_rng=False,
    )
    torch.testing.assert_close(restored.weight, model.module.weight)
    assert state["step"] == 1
    assert state["batches_in_epoch"] == 2
    if context.rank == 0:
        print("tiny DDP integration passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
