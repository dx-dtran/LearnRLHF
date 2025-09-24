import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import data
import gpt
import train_ppo
import train_rm
import train_sft


def _build_tiny_tokenizer():
    class TinyEncoder:
        def __init__(self) -> None:
            self.n_vocab = 12

        def encode(self, text: str) -> list[int]:
            if not text:
                return []
            return [3 + (byte % 5) for byte in text.encode("utf-8")]

    return data.TokenizerBundle(encoder=TinyEncoder(), bos=0, eos=1, pad=2)


@pytest.fixture
def tiny_training_setup(monkeypatch):
    def tiny_config(**overrides) -> gpt.GPTConfig:
        config = gpt.GPTConfig(
            vocab_size=12,
            block_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    monkeypatch.setattr(data, "build_tokenizer", _build_tiny_tokenizer)
    for module in (train_sft, train_rm, train_ppo):
        monkeypatch.setattr(module, "GPTConfig", tiny_config)

    return tiny_config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_supervised_file(tmp_path: Path) -> Path:
    path = tmp_path / "sft.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "Hello", "chosen": "World"},
            {"prompt": "How", "chosen": "are you"},
        ],
    )
    return path


def _make_preferences_file(tmp_path: Path) -> Path:
    path = tmp_path / "preferences.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "Question", "chosen": "Good", "rejected": "Bad"},
            {"prompt": "Another", "chosen": "Yes", "rejected": "No"},
            {"prompt": "More", "chosen": "Up", "rejected": "Down"},
        ],
    )
    return path


def test_train_sft_saves_checkpoint(tmp_path, tiny_training_setup):
    torch.manual_seed(0)
    data_path = _make_supervised_file(tmp_path)
    out_dir = tmp_path / "weights"

    checkpoint = train_sft.train_sft(
        str(data_path),
        out_dir=str(out_dir),
        batch_size=2,
        epochs=1,
        grad_accumulation_steps=2,
        device=torch.device("cpu"),
    )

    assert Path(checkpoint).exists()
    state = torch.load(checkpoint, map_location="cpu")
    assert any(key.startswith("token_embed") for key in state)


def test_train_sft_invalid_accumulation(tmp_path, tiny_training_setup):
    data_path = _make_supervised_file(tmp_path)
    out_dir = tmp_path / "weights"

    with pytest.raises(ValueError):
        train_sft.train_sft(
            str(data_path),
            out_dir=str(out_dir),
            batch_size=2,
            epochs=1,
            grad_accumulation_steps=0,
            device=torch.device("cpu"),
        )


def test_train_sft_missing_file(tmp_path, tiny_training_setup):
    with pytest.raises(FileNotFoundError):
        train_sft.train_sft(
            str(tmp_path / "missing.jsonl"),
            out_dir=str(tmp_path / "weights"),
            device=torch.device("cpu"),
        )


def test_train_reward_model_runs(tmp_path, tiny_training_setup):
    torch.manual_seed(1)
    preference_path = _make_preferences_file(tmp_path)
    out_path = tmp_path / "reward.pt"

    checkpoint = train_rm.train_reward_model(
        str(preference_path),
        out_path=str(out_path),
        batch_size=2,
        epochs=1,
        grad_accumulation_steps=2,
        init_path=None,
        device=torch.device("cpu"),
    )

    assert Path(checkpoint).exists()
    state = torch.load(checkpoint, map_location="cpu")
    assert "score.weight" in state


def test_train_reward_model_invalid_accumulation(tmp_path, tiny_training_setup):
    preference_path = _make_preferences_file(tmp_path)

    with pytest.raises(ValueError):
        train_rm.train_reward_model(
            str(preference_path),
            out_path=str(tmp_path / "reward.pt"),
            grad_accumulation_steps=0,
            init_path=None,
            device=torch.device("cpu"),
        )


def test_train_reward_model_missing_files(tmp_path, tiny_training_setup):
    with pytest.raises(FileNotFoundError):
        train_rm.train_reward_model(
            str(tmp_path / "missing.jsonl"),
            out_path=str(tmp_path / "reward.pt"),
            init_path=None,
            device=torch.device("cpu"),
        )

    preference_path = _make_preferences_file(tmp_path)
    with pytest.raises(FileNotFoundError):
        train_rm.train_reward_model(
            str(preference_path),
            out_path=str(tmp_path / "reward.pt"),
            init_path=str(tmp_path / "nope.pt"),
            device=torch.device("cpu"),
        )


def test_train_ppo_runs(tmp_path, tiny_training_setup):
    torch.manual_seed(2)
    preference_path = _make_preferences_file(tmp_path)
    reward_path = tmp_path / "reward.pt"
    out_path = tmp_path / "ppo.pt"

    reward_model = train_rm.ScalarHead(tiny_training_setup())
    torch.save(reward_model.state_dict(), reward_path)

    checkpoint = train_ppo.train_ppo(
        str(preference_path),
        reward_path=str(reward_path),
        policy_init=None,
        out_path=str(out_path),
        batch_size=2,
        epochs=1,
        max_new_tokens=2,
        device=torch.device("cpu"),
    )

    assert Path(checkpoint).exists()
    state = torch.load(checkpoint, map_location="cpu")
    assert "token_embed.weight" in state


def test_train_ppo_missing_files(tmp_path, tiny_training_setup):
    reward_model = train_rm.ScalarHead(tiny_training_setup())
    reward_path = tmp_path / "reward.pt"
    torch.save(reward_model.state_dict(), reward_path)

    with pytest.raises(FileNotFoundError):
        train_ppo.train_ppo(
            str(tmp_path / "missing.jsonl"),
            reward_path=str(reward_path),
        )

    preference_path = _make_preferences_file(tmp_path)
    with pytest.raises(FileNotFoundError):
        train_ppo.train_ppo(
            str(preference_path),
            reward_path=str(tmp_path / "missing_reward.pt"),
        )


def test_train_ppo_missing_policy_init(tmp_path, tiny_training_setup):
    preference_path = _make_preferences_file(tmp_path)
    reward_model = train_rm.ScalarHead(tiny_training_setup())
    reward_path = tmp_path / "reward.pt"
    torch.save(reward_model.state_dict(), reward_path)

    with pytest.raises(FileNotFoundError):
        train_ppo.train_ppo(
            str(preference_path),
            reward_path=str(reward_path),
            policy_init=str(tmp_path / "missing_policy.pt"),
        )
