"""
tokenizer.py — tiktoken wrapper + chat template helpers.

Modules this file covers:
    0.2  inspect HH token lengths
    2.1  chat formatter + loss mask

We use the stock GPT-2 BPE from `tiktoken`. The chat tags `<|im_start|>` and `<|im_end|>`
are NOT special tokens in GPT-2 BPE — we deliberately encode them as their literal UTF-8
bytes so that no vocab surgery is needed. This matches the data format in the original
jsonl and keeps things transparent.

Public API (what the rest of the repo imports):
    enc                       -> a module-level tiktoken.Encoding
    IM_START, IM_END, EOT     -> strings / token ids
    format_chat(turns)        -> str
    format_prompt(turns)      -> str (trailing "<|im_start|>assistant\\n" cue)
    build_sft_example(turns)  -> (input_ids, loss_mask)  [Problem 2.1]
    encode(text), decode(ids) -> passthrough helpers
"""

from typing import List, Tuple

import tiktoken


# -------------------------------------------------------------------------------------
# tiktoken encoder
# -------------------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# GPT-2 BPE's real special token is <|endoftext|> (id 50256).
EOT_ID = enc.eot_token  # 50256


def encode(text: str) -> List[int]:
    """Plain BPE encode — no special-token handling."""
    return enc.encode(text, disallowed_special=())


def decode(ids: List[int]) -> str:
    return enc.decode(ids)


# -------------------------------------------------------------------------------------
# Chat template
# -------------------------------------------------------------------------------------
# A "turn" is a dict {"role": "user"|"assistant", "content": str}.
# Format (trailing newline after content, then <|im_end|>\n):
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
    """
    Render turns but leave an open assistant cue for generation:

        <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    """
    return format_chat(turns) + f"{IM_START}assistant\n"


# -------------------------------------------------------------------------------------
# Problem 2.1 — Build SFT example with a proper loss mask
# -------------------------------------------------------------------------------------

def build_sft_example(turns: List[dict]) -> Tuple[List[int], List[int]]:
    """
    Tokenize a full dialogue and return (input_ids, loss_mask) of equal length.

    `loss_mask[i] == 1` iff token `i` is part of an assistant response (including the
    assistant's trailing `<|im_end|>\\n` tokens). User tokens and chat scaffolding
    (`<|im_start|>user\\n`, role tags, etc.) must be masked to 0.

    Why this matters:
        The old code in this repo trained on *every* token — meaning GPT-2 was rewarded
        for predicting user messages too. That's wrong: we want a model that predicts the
        assistant's text given the user's, not one that predicts the user's messages.

    Strategy (one easy approach; feel free to invent your own):
        1. Start with the empty output lists.
        2. For each turn:
             a. Tokenize the SCAFFOLD: f"{IM_START}{role}\\n"     -> mask = 0
             b. Tokenize the CONTENT:  f"{content}{IM_END}\\n"    -> mask = 1 if assistant
                                                                     else 0
             c. Append tokens + mask bits to the output.
        3. Return both lists, same length.

    TODO(2.1): implement.

    Hints:
        - Encode small chunks separately and concatenate. Do NOT try to find roles by
          searching for `<|im_start|>` in an already-encoded id stream — BPE makes that
          unreliable.
        - Your unit test (tests/test_data.py) will check: (a) lengths match; (b) the
          positions of 1s in loss_mask correspond only to assistant content + its
          trailing <|im_end|>\\n.
    """
    raise NotImplementedError("TODO(2.1): implement build_sft_example")


# -------------------------------------------------------------------------------------
# Problem 0.2 — scratch helper you can use in notes/00-data.md
# -------------------------------------------------------------------------------------

def count_tokens_in_chat(turns: List[dict]) -> int:
    return len(encode(format_chat(turns)))