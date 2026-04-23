"""
eval.py — side-by-side generation for qualitative evaluation.

Module 2.5 / 6.1. Loads 1–3 checkpoints (base, SFT, RLHF) and generates responses for
the same held-out prompts, emitting a markdown table to stdout or a file.

Usage (CLI sketch; implement the argparse in Module 6.1):
    python eval.py --models base,sft,rlhf --n 20 --out notes/06-eval.md
"""

import argparse
from typing import List

import torch

from config import GPTConfig
from model import GPT, load_gpt2_from_hf
from tokenizer import decode, encode, format_prompt
from data_hh import download_hh, parse_hh_dialogue, split_prompt_and_last_response


# TODO(2.5 / 6.1):
#   def load_model(kind: str) -> GPT:
#       if kind == "base":    load_gpt2_from_hf
#       elif kind == "sft":   load sft.pt
#       elif kind == "rlhf":  load rlhf.pt
#
#   def sample_prompts(n: int) -> list[list[dict]]:
#       pull n random held-out HH rows, parse, return their prompt-only turn lists
#
#   def run(model, prompts, max_new=256, temperature=0.8, top_p=0.9) -> list[str]:
#       for each prompt: encode, model.generate, decode until EOS
#
#   def render_markdown(prompts, model_outputs: dict[str, list[str]]) -> str:
#       one row per prompt, one column per model
#
#   if __name__ == "__main__": argparse → run → write


def main():
    raise NotImplementedError("TODO(2.5 / 6.1): eval.main")


if __name__ == "__main__":
    main()