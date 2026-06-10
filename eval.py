"""
eval.py — side-by-side generation: base GPT-2 vs SFT vs RLHF.

Fully provided. Part 5 of the crash course is running it and judging the outputs:

    python eval.py --models base,sft,rlhf --n 20 --out notes/06-eval.md

Emits a markdown table, one row per held-out prompt, one column per model. Blind
yourself to the column order before judging if you want an honest win rate.
"""

import argparse
import random

import torch

from config import GPTConfig
from data_hh import download_hh, parse_hh_dialogue, split_prompt_and_last_response
from model import GPT, load_gpt2_from_hf
from tokenizer import EOT_ID, IM_END, decode, encode, format_prompt


def load_model(kind: str, device) -> GPT:
    model = GPT(GPTConfig())
    if kind == "base":
        load_gpt2_from_hf(model, "gpt2")
    else:
        path = {"sft": "sft.pt", "rlhf": "rlhf.pt"}[kind]
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def sample_prompts(n: int, seed: int = 0, max_tokens: int = 512) -> list:
    rows = download_hh("test")
    rng = random.Random(seed)
    rng.shuffle(rows)
    prompts = []
    for row in rows:
        turns = parse_hh_dialogue(row["chosen"])
        if not turns or turns[-1]["role"] != "assistant":
            continue
        prompt_turns, _ = split_prompt_and_last_response(turns)
        if len(encode(format_prompt(prompt_turns))) > max_tokens:
            continue
        prompts.append(prompt_turns)
        if len(prompts) == n:
            break
    return prompts


@torch.no_grad()
def run(model: GPT, prompts: list, device, max_new: int = 256,
        temperature: float = 0.8, top_p: float = 0.9) -> list:
    outs = []
    for turns in prompts:
        ids = torch.tensor([encode(format_prompt(turns))], device=device)
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=temperature,
            top_p=top_p, eos_token_id=EOT_ID,
        )
        text = decode(out[0, ids.size(1):].tolist())
        # cut at the first end-of-turn or end-of-text the model emits
        for stop in (IM_END, "<|endoftext|>"):
            if stop in text:
                text = text.split(stop)[0]
        outs.append(text.strip())
    return outs


def render_markdown(prompts: list, outputs: dict) -> str:
    def cell(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", "<br>")

    names = list(outputs)
    lines = ["| prompt | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for i, turns in enumerate(prompts):
        prompt_text = " / ".join(f"{t['role']}: {t['content']}" for t in turns)
        row = [cell(prompt_text)] + [cell(outputs[n][i]) for n in names]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,sft", help="comma list: base,sft,rlhf")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", default=None, help="write markdown here (default stdout)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = sample_prompts(args.n, seed=args.seed)

    outputs = {}
    for kind in args.models.split(","):
        kind = kind.strip()
        print(f"generating with {kind}...")
        model = load_model(kind, device)
        outputs[kind] = run(model, prompts, device, max_new=args.max_new)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    md = render_markdown(prompts, outputs)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
