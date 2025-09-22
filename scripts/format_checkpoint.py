"""Utility helpers for converting legacy GPT-2 checkpoints.

Run this script once on weights that were saved with an older layout or with
transpose requirements. The training code now assumes checkpoints are already
in the "nanogpt-style" layout, so we keep the conversion logic here instead of
sprinkling it across the trainers.
"""

import argparse
from collections import OrderedDict
from typing import MutableMapping

import torch

_TRANSPOSE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _should_transpose(
    key: str, value: torch.Tensor, state_dict: MutableMapping[str, torch.Tensor]
) -> bool:
    if value.ndim != 2 or not any(key.endswith(suffix) for suffix in _TRANSPOSE_SUFFIXES):
        return False

    bias_key = key[:-6] + "bias"
    bias = state_dict.get(bias_key)
    if bias is None or bias.ndim != 1:
        return False

    expected_out = bias.shape[0]
    if value.shape[0] == expected_out:
        return False
    if value.shape[1] == expected_out:
        return True
    return False


def transpose_legacy_state(
    state_dict: MutableMapping[str, torch.Tensor]
) -> MutableMapping[str, torch.Tensor]:
    keys_to_transpose = {
        key
        for key, value in state_dict.items()
        if _should_transpose(key, value, state_dict)
    }

    if not keys_to_transpose:
        return state_dict

    if isinstance(state_dict, OrderedDict):
        converted: MutableMapping[str, torch.Tensor] = OrderedDict()
    else:
        converted = {}

    for key, value in state_dict.items():
        if key in keys_to_transpose:
            converted[key] = value.transpose(0, 1)
        else:
            converted[key] = value
    return converted


def add_prefix(state_dict: MutableMapping[str, torch.Tensor], prefix: str) -> None:
    if not prefix:
        return
    for key in list(state_dict.keys()):
        state_dict[f"{prefix}{key}"] = state_dict.pop(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to the legacy checkpoint")
    parser.add_argument("output", type=str, help="Where to write the converted checkpoint")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help=(
            "Optional prefix to add to every key. Useful for preparing policy weights "
            "for modules such as ScalarHead.body."
        ),
    )
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu")
    state = transpose_legacy_state(state)
    if args.prefix:
        add_prefix(state, args.prefix)
    torch.save(state, args.output)


if __name__ == "__main__":
    main()
