import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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


def test_train_sft_reduces_loss(tmp_path, tiny_training_setup):
    torch.manual_seed(0)
    data_path = _make_supervised_file(tmp_path)
    out_dir = tmp_path / "weights"
    metrics: list[dict[str, float]] = []

    train_sft.train_sft(
        str(data_path),
        out_dir=str(out_dir),
        batch_size=2,
        epochs=3,
        lr=5e-3,
        grad_accumulation_steps=1,
        device=torch.device("cpu"),
        metrics=metrics,
    )

    assert metrics and metrics[0]["epoch"] == 0
    final = metrics[-1]
    initial = metrics[0]
    assert final["loss"] < initial["loss"]
    assert final["token_accuracy"] >= initial["token_accuracy"]


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


def test_train_reward_model_reduces_loss(tmp_path, tiny_training_setup):
    torch.manual_seed(1)
    preference_path = _make_preferences_file(tmp_path)
    out_path = tmp_path / "reward.pt"
    metrics: list[dict[str, float]] = []

    train_rm.train_reward_model(
        str(preference_path),
        out_path=str(out_path),
        batch_size=2,
        epochs=4,
        lr=1e-3,
        grad_accumulation_steps=1,
        init_path=None,
        device=torch.device("cpu"),
        metrics=metrics,
    )

    assert metrics and metrics[0]["epoch"] == 0
    final = metrics[-1]
    initial = metrics[0]
    assert final["loss"] < initial["loss"]
    assert final["preference_accuracy"] >= initial["preference_accuracy"]


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


def test_ppo_train_step_improves_policy(tmp_path, tiny_training_setup):
    torch.manual_seed(2)
    preference_path = _make_preferences_file(tmp_path)
    config = tiny_training_setup()

    dataset = data.PreferenceDataset(preference_path, block_size=config.block_size)
    tokenizer_bundle = dataset.tokenizer_bundle
    row = dataset[0]

    def build_sample(policy_model, trainer_obj):
        prompt_mask = row["prompt_mask"]
        chosen = row["chosen"]
        chosen_mask = row["chosen_mask"].float()

        prompt_length = int(prompt_mask.sum().item())
        chosen_length = int(row["chosen_mask"].sum().item())
        response_length = max(chosen_length - prompt_length, 0)
        assert response_length > 0, "synthetic preference row must contain a response"

        responses = torch.full((1, response_length), tokenizer_bundle.pad, dtype=torch.long)
        responses_mask = torch.zeros((1, response_length), dtype=torch.float32)
        actual_response = chosen[prompt_length:prompt_length + response_length]
        responses[0, :response_length] = actual_response
        responses_mask[0, :response_length] = 1.0

        full = chosen.unsqueeze(0)
        full_mask = chosen_mask.unsqueeze(0)

        response_mask = torch.zeros_like(full_mask)
        response_mask[0, prompt_length:prompt_length + response_length] = 1.0

        prompt_lengths = torch.tensor([prompt_length], dtype=torch.long)
        response_lengths = torch.tensor([response_length], dtype=torch.long)

        hidden = policy_model.transform(full, attention_mask=full_mask)
        logits = policy_model.head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = train_ppo.gather_response_log_probs(
            log_probs, responses, prompt_lengths, response_lengths
        )
        old_values = trainer_obj.compute_values(hidden, full_mask)

        return {
            "full": full,
            "full_mask": full_mask,
            "responses": responses,
            "responses_mask": responses_mask,
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "old_log_probs": old_log_probs.detach(),
            "old_values": old_values.detach(),
            "response_mask": response_mask,
        }

    # Value head should move toward the reward when policy gradients are disabled.
    value_policy = gpt.GPT(config)
    value_reference = gpt.GPT(config)
    value_reference.load_state_dict(value_policy.state_dict())
    value_head = train_ppo.ValueHead(config.n_embd)
    value_policy.add_module("value_head", value_head)
    reward_model = train_rm.ScalarHead(config)

    value_trainer = train_ppo.PPOTrainer(
        value_policy,
        value_reference,
        value_head,
        reward_model,
        pad_token=tokenizer_bundle.pad,
        lr=1e-3,
    )

    for name, param in value_policy.named_parameters():
        if not name.startswith("value_head"):
            param.requires_grad_(False)

    value_sample = build_sample(value_policy, value_trainer)
    value_sample["advantages"] = torch.zeros_like(value_sample["old_values"])
    value_sample["rewards"] = value_sample["old_values"] + 1.0

    value_trainer.train_step(value_sample)
    with torch.no_grad():
        new_hidden = value_policy.transform(
            value_sample["full"], attention_mask=value_sample["full_mask"]
        )
        new_values = value_trainer.compute_values(new_hidden, value_sample["full_mask"])

    old_error = (value_sample["old_values"] - value_sample["rewards"]).abs().mean()
    new_error = (new_values - value_sample["rewards"]).abs().mean()
    assert float(new_error.item()) < float(old_error.item())

    # Policy loss should decrease when value head updates are disabled.
    policy_model = gpt.GPT(config)
    policy_reference = gpt.GPT(config)
    policy_reference.load_state_dict(policy_model.state_dict())
    policy_value_head = train_ppo.ValueHead(config.n_embd)
    policy_model.add_module("value_head", policy_value_head)

    policy_trainer = train_ppo.PPOTrainer(
        policy_model,
        policy_reference,
        policy_value_head,
        reward_model,
        pad_token=tokenizer_bundle.pad,
        lr=1e-3,
    )

    for param in policy_value_head.parameters():
        param.requires_grad_(False)

    policy_sample = build_sample(policy_model, policy_trainer)
    policy_sample["advantages"] = torch.ones_like(policy_sample["old_values"])
    policy_sample["rewards"] = policy_sample["old_values"]

    def compute_policy_loss(sample: dict[str, torch.Tensor]) -> float:
        hidden = policy_model.transform(sample["full"], attention_mask=sample["full_mask"])
        logits = policy_model.head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            ref_logits, _ = policy_reference(sample["full"], attention_mask=sample["full_mask"])
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_loss, _, _ = policy_trainer.compute_policy_losses(
            log_probs,
            ref_log_probs,
            sample["responses"],
            sample["prompt_lengths"],
            sample["response_lengths"],
            sample["responses_mask"],
            sample["response_mask"],
            sample["advantages"],
            sample["old_log_probs"],
        )
        return float(policy_loss.item())

    before = compute_policy_loss(policy_sample)
    policy_trainer.train_step(policy_sample)
    after = compute_policy_loss(policy_sample)
    assert after < before


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
