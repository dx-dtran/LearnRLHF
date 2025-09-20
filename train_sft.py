import argparse
import math
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from data import SupervisedDataset, collate_supervised, build_tokenizer
from gpt import GPT, GPTConfig


def cosine_schedule(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def setup_distributed() -> tuple[int, int, int | None]:
    if "RANK" not in os.environ:
        return 0, 1, None
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    return rank, world, local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--out", type=str, default="weights")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--init", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    rank, world, local_rank = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and local_rank is not None:
        device = torch.device("cuda", local_rank)

    config = GPTConfig(dropout=args.dropout)
    model = GPT(config).to(device)

    if args.init:
        state = torch.load(args.init, map_location="cpu")
        model.load_state_dict(state, strict=False)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if local_rank is not None else None
        )

    dataset = SupervisedDataset(args.data, block_size=config.block_size)
    bundle = build_tokenizer()
    sampler = (
        DistributedSampler(dataset, shuffle=True) if world > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=lambda batch: collate_supervised(batch, bundle.pad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    total_steps = args.epochs * math.ceil(len(loader) / args.accum)
    step = 0

    model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits, loss = model(inputs, targets=targets, attention_mask=mask)
            loss = loss / args.accum
            loss.backward()

            if (step + 1) % args.accum == 0:
                lr = cosine_schedule(step // args.accum, total_steps, args.lr, args.warmup)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            step += 1

        if rank == 0:
            os.makedirs(args.out, exist_ok=True)
            fname = os.path.join(args.out, f"sft_epoch_{epoch+1}.pt")
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, fname)


if __name__ == "__main__":
    main()
