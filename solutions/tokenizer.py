"""
tokenizer.py — tiktoken wrapper + chat template + SFT loss mask.

Part 2 of the crash course. Your job:

    [FILL 2.1]  build_sft_example — tokenize a dialogue and build the loss mask that
                is 1 only on assistant tokens. This is THE classic SFT bug; an early
                version of this repo trained on user turns too.

GPT-2 BPE comes from tiktoken. The chat tags <|im_start|> / <|im_end|> are NOT
special tokens — they are encoded as their literal UTF-8 bytes and BPE splits them
however it likes. That is fine: we only ever tokenize chunk-by-chunk, never search
for tag ids inside an encoded stream.

The one real special token is <|endoftext|> (id 50256). SFT appends it after the
final assistant turn so the model learns to stop; generation and PPO rollouts use it
as the EOS id.

Offline fallback: if tiktoken cannot download its BPE files (no network), we fall
back to a plain byte-level encoder (ids 0..255, EOT=256) so the unit tests still run.
Real training needs the real encoding; you'll see a loud warning if the fallback is
active.
"""

import sys
from typing import List, Tuple


class _ByteFallbackEncoding:
    """Byte-level stand-in for tiktoken's GPT-2 encoding (tests/offline only)."""

    name = "byte-fallback"
    eot_token = 256

    def encode(self, text: str, disallowed_special=()) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def _load_encoding():
    import tiktoken

    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as e:  # no network: BPE files live on a remote host
        print(
            f"[tokenizer.py] WARNING: could not load GPT-2 BPE ({e!r}); "
            "using byte-level fallback. Fine for unit tests, NOT for training.",
            file=sys.stderr,
        )
        return _ByteFallbackEncoding()


enc = _load_encoding()

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOT_ID = enc.eot_token  # <|endoftext|>, 50256 with real GPT-2 BPE


def encode(text: str) -> List[int]:
    """Plain BPE encode — no special-token handling."""
    return enc.encode(text, disallowed_special=())


def decode(ids: List[int]) -> str:
    return enc.decode(ids)


# -------------------------------------------------------------------------------------
# Chat template
# -------------------------------------------------------------------------------------
# A "turn" is a dict {"role": "user"|"assistant", "content": str}. Rendered as:
#
#   <|im_start|>user
#   hello<|im_end|>
#   <|im_start|>assistant
#   hi!<|im_end|>
#


def format_chat(turns: List[dict]) -> str:
    """Render a list of turns into the full chat template string."""
    parts = []
    for t in turns:
        parts.append(f"{IM_START}{t['role']}\n{t['content']}{IM_END}\n")
    return "".join(parts)


def format_prompt(turns: List[dict]) -> str:
    """Render turns and leave an open assistant cue for generation."""
    return format_chat(turns) + f"{IM_START}assistant\n"


# -------------------------------------------------------------------------------------
# [FILL 2.1] — SFT example with a per-token loss mask
# -------------------------------------------------------------------------------------


def build_sft_example(turns: List[dict]) -> Tuple[List[int], List[int]]:
    """
    Tokenize a full dialogue and return (input_ids, loss_mask), equal length.

    loss_mask[i] == 1 iff token i is assistant CONTENT (including the assistant's
    trailing "<|im_end|>\\n"). Everything else — user content, both roles' scaffold
    "<|im_start|>{role}\\n" — gets 0. The final assistant turn is followed by one
    <|endoftext|> token with mask 1 (the model must learn to stop).

    Strategy: tokenize the dialogue chunk by chunk and assign each chunk's mask bit
    as you go. Per turn:
        scaffold f"{IM_START}{role}\\n"   -> mask 0
        content  f"{content}{IM_END}\\n"  -> mask 1 if role == "assistant" else 0
    then append EOT_ID with mask 1 after the last turn if it is an assistant turn.

    Do NOT search for tag ids in an already-encoded stream; BPE merges across
    boundaries make that unreliable. Encoding chunks separately is the whole trick.
    """
    input_ids: List[int] = []
    loss_mask: List[int] = []
    for t in turns:
        scaffold = encode(f"{IM_START}{t['role']}\n")
        input_ids += scaffold
        loss_mask += [0] * len(scaffold)

        content = encode(f"{t['content']}{IM_END}\n")
        bit = 1 if t["role"] == "assistant" else 0
        input_ids += content
        loss_mask += [bit] * len(content)

    if turns and turns[-1]["role"] == "assistant":
        input_ids.append(EOT_ID)
        loss_mask.append(1)
    return input_ids, loss_mask


def count_tokens_in_chat(turns: List[dict]) -> int:
    return len(encode(format_chat(turns)))
