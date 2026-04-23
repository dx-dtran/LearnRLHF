"""
RLHF PPO Implementation for GPT-2 Small
Based on the InstructGPT paper and RLHF methodology.

This module implements Proximal Policy Optimization (PPO) for Reinforcement Learning 
from Human Feedback (RLHF) on GPT-2 small. The implementation includes:

1. Reward Model (RM) - trained on human preferences
2. PPO with clipped objective and KL penalty
3. Value function for advantage estimation
4. Complete RLHF training loop

Key components:
- State: Text prompt/context
- Action: Generated token sequence
- Reward: Scalar score from reward model based on human preferences
- Policy: GPT-2 model being fine-tuned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Dict, List, Tuple, Optional
import math
import copy

# Import existing GPT implementation
from gpt import GPT, GPTConfig


class RewardModel(nn.Module):
    """Reward model trained on human preferences."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Use the existing GPT implementation as backbone
        self.backbone = GPT(config)
        # Add reward head that outputs scalar score
        self.reward_head = nn.Linear(config.n_embd, 1)

        # Initialize reward head with small weights
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward score for input sequence.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Reward scores of shape (batch_size,)

        Mathematical operations:
        1. Get final hidden states from backbone: h = GPT2Backbone(input_ids)
        2. Pool sequence representations (take last token)
        3. Apply reward head: reward = RewardHead(pooled_h)
        4. Return scalar reward per sequence
        """
        # Forward through backbone to get hidden states directly
        text_embeds = self.backbone.wte(input_ids)
        batch_size, text_len, _ = text_embeds.size()
        pos_ids = torch.arange(0, text_len, dtype=torch.long, device=text_embeds.device)
        pos_emb = (
            self.backbone.wpe(pos_ids).unsqueeze(0).expand(batch_size, text_len, -1)
        )
        text_embeds = text_embeds + pos_emb

        batch_size, seq_length, _ = text_embeds.shape
        causal_mask = self.backbone._create_causal_mask(seq_length)

        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask & attention_mask_expanded
        else:
            combined_mask = causal_mask

        hidden_states, _ = self.backbone._forward_transformer_blocks(
            text_embeds, mask=combined_mask
        )

        # Pool by taking the last token representation
        if attention_mask is not None:
            # Find the last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
        else:
            # Take the last token
            pooled = hidden_states[:, -1, :]

        # Apply reward head to get scalar scores
        rewards = self.reward_head(pooled).squeeze(-1)  # (batch_size,)

        return rewards

    def compute_pairwise_loss(
        self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss for reward model training.

        Args:
            chosen_rewards: Rewards for preferred outputs, shape (batch_size,)
            rejected_rewards: Rewards for rejected outputs, shape (batch_size,)

        Returns:
            Pairwise ranking loss scalar

        Mathematical operations:
        1. Compute preference probability: p = sigmoid(chosen_rewards - rejected_rewards)
        2. Compute cross-entropy loss: loss = -log(p)
        3. This encourages chosen_rewards > rejected_rewards
        """
        # Compute difference in rewards
        reward_diff = chosen_rewards - rejected_rewards

        # Apply sigmoid to get preference probability
        # This represents P(chosen > rejected)
        pref_prob = torch.sigmoid(reward_diff)

        # Compute negative log likelihood loss
        # We want to maximize log(pref_prob), so minimize -log(pref_prob)
        loss = -torch.log(
            pref_prob + 1e-8
        ).mean()  # Add epsilon for numerical stability

        return loss


class ValueFunction(nn.Module):
    """Value function for advantage estimation in PPO."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Use the existing GPT implementation as backbone
        self.backbone = GPT(config)
        # Add value head that outputs scalar value estimate
        self.value_head = nn.Linear(config.n_embd, 1)

        # Initialize value head with small weights
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate value (expected reward) for input states.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Value estimates of shape (batch_size,)

        Mathematical operations:
        1. Get hidden states from backbone: h = GPT2Backbone(input_ids)
        2. Pool sequence representations
        3. Apply value head: value = ValueHead(pooled_h)
        4. Return scalar value estimate per sequence
        """
        # Forward through backbone to get hidden states directly
        text_embeds = self.backbone.wte(input_ids)
        batch_size, text_len, _ = text_embeds.size()
        pos_ids = torch.arange(0, text_len, dtype=torch.long, device=text_embeds.device)
        pos_emb = (
            self.backbone.wpe(pos_ids).unsqueeze(0).expand(batch_size, text_len, -1)
        )
        text_embeds = text_embeds + pos_emb

        batch_size, seq_length, _ = text_embeds.shape
        causal_mask = self.backbone._create_causal_mask(seq_length)

        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask & attention_mask_expanded
        else:
            combined_mask = causal_mask

        hidden_states, _ = self.backbone._forward_transformer_blocks(
            text_embeds, mask=combined_mask
        )

        # Pool by taking the last token representation
        if attention_mask is not None:
            # Find the last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
        else:
            # Take the last token
            pooled = hidden_states[:, -1, :]

        # Apply value head to get scalar value estimates
        values = self.value_head(pooled).squeeze(-1)  # (batch_size,)

        return values


class PPOTrainer:
    """PPO trainer for RLHF."""

    def __init__(
        self,
        policy_model: GPT,
        value_model: ValueFunction,
        reward_model: RewardModel,
        reference_model: GPT,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.02,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        learning_rate: float = 1e-5,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_model: The policy model being trained (active GPT-2)
            value_model: Value function for advantage estimation
            reward_model: Frozen reward model for scoring outputs
            reference_model: Frozen reference policy (original SFT model)
            clip_epsilon: PPO clipping parameter (typically 0.1-0.3)
            kl_coeff: KL divergence penalty coefficient
            entropy_coeff: Entropy bonus coefficient
            value_coeff: Value function loss coefficient
            learning_rate: Learning rate for optimization
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.reference_model = reference_model

        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        # Freeze reference and reward models
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Set up optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            value_model.parameters(), lr=learning_rate
        )

    def compute_rewards(
        self, prompts: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards for prompt-response pairs using reward model.

        Args:
            prompts: Prompt token IDs of shape (batch_size, prompt_len)
            responses: Response token IDs of shape (batch_size, response_len)

        Returns:
            Rewards of shape (batch_size,)

        Mathematical operations:
        1. Concatenate prompts and responses: full_seq = concat(prompts, responses)
        2. Get reward from reward model: rewards = reward_model(full_seq)
        3. Return rewards (one scalar per sequence)
        """
        # Concatenate prompts and responses along sequence dimension
        full_sequences = torch.cat([prompts, responses], dim=1)

        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones_like(full_sequences)

        # Get rewards from reward model
        with torch.no_grad():
            rewards = self.reward_model(full_sequences, attention_mask=attention_mask)

        return rewards

    def compute_kl_penalty(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference model.

        Args:
            logits: Current policy logits of shape (batch_size, seq_len, vocab_size)
            ref_logits: Reference policy logits of shape (batch_size, seq_len, vocab_size)
            attention_mask: Mask for valid tokens of shape (batch_size, seq_len)

        Returns:
            KL penalty scalar

        Mathematical operations:
        1. Convert logits to log probabilities: log_p = log_softmax(logits)
        2. Convert reference to log probabilities: log_p_ref = log_softmax(ref_logits)
        3. Compute KL divergence: kl = exp(log_p) * (log_p - log_p_ref)
        4. Sum over vocabulary: kl_per_token = sum(kl, dim=-1)
        5. Mask invalid tokens and average: kl_penalty = mean(kl_per_token * attention_mask)
        """
        # Convert to log probabilities for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Convert to probabilities for KL computation
        probs = torch.exp(log_probs)

        # Compute KL divergence: KL(p||q) = sum(p * log(p/q))
        kl_per_token = torch.sum(probs * (log_probs - ref_log_probs), dim=-1)

        # Apply attention mask and compute mean
        masked_kl = kl_per_token * attention_mask.float()
        kl_penalty = masked_kl.sum() / attention_mask.float().sum().clamp(min=1)

        return kl_penalty

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages using reward and value estimates.

        Args:
            rewards: Reward scores of shape (batch_size,)
            values: Value estimates of shape (batch_size,)

        Returns:
            Advantages of shape (batch_size,)

        Mathematical operations:
        1. Compute advantages: advantages = rewards - values
        2. Normalize: advantages = (advantages - mean) / (std + epsilon)
        3. This measures how much better/worse the outcome was vs. expectation
        """
        # Compute raw advantages
        advantages = rewards - values

        # Normalize advantages for more stable training
        mean_adv = advantages.mean()
        std_adv = advantages.std()
        normalized_advantages = (advantages - mean_adv) / (std_adv + 1e-8)

        return normalized_advantages

    def compute_ppo_loss(
        self,
        logits: torch.Tensor,
        old_logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO clipped objective loss.

        Args:
            logits: New policy logits of shape (batch_size, seq_len, vocab_size)
            old_logits: Old policy logits of shape (batch_size, seq_len, vocab_size)
            actions: Token actions of shape (batch_size, seq_len)
            advantages: Advantage estimates of shape (batch_size,)
            attention_mask: Mask for valid tokens of shape (batch_size, seq_len)

        Returns:
            PPO loss scalar

        Mathematical operations:
        1. Compute log probabilities: log_probs = log_softmax(logits)[actions]
        2. Compute old log probabilities: old_log_probs = log_softmax(old_logits)[actions]
        3. Compute probability ratio: ratio = exp(log_probs - old_log_probs)
        4. Compute clipped ratio: clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
        5. Compute surrogate objectives: obj1 = ratio * advantages, obj2 = clipped_ratio * advantages
        6. Take minimum: surrogate_loss = -min(obj1, obj2)
        7. Mask invalid tokens and average over sequence length
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = F.log_softmax(old_logits, dim=-1)

        # Extract log probabilities for actual actions taken
        # Gather the log probs for the actions that were actually taken
        log_probs_actions = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_log_probs_actions = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(
            -1
        )

        # Compute probability ratios
        ratios = torch.exp(log_probs_actions - old_log_probs_actions)

        # Expand advantages to match sequence dimension
        # advantages has shape (batch_size,), we need (batch_size, seq_len)
        expanded_advantages = advantages.unsqueeze(1).expand_as(ratios)

        # Compute surrogate objectives
        surr1 = ratios * expanded_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * expanded_advantages
        )

        # Take minimum (we want to maximize, so we'll negate later)
        surrogate_obj = torch.min(surr1, surr2)

        # Apply attention mask and compute mean
        masked_obj = surrogate_obj * attention_mask.float()
        ppo_loss = -masked_obj.sum() / attention_mask.float().sum().clamp(min=1)

        return ppo_loss

    def compute_value_loss(
        self, predicted_values: torch.Tensor, target_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss.

        Args:
            predicted_values: Value predictions of shape (batch_size,)
            target_values: Target values (rewards) of shape (batch_size,)

        Returns:
            Value loss scalar

        Mathematical operations:
        1. Compute MSE loss: loss = mean((predicted_values - target_values)^2)
        2. This trains the value function to better predict rewards
        """
        # Compute mean squared error loss
        value_loss = F.mse_loss(predicted_values, target_values)
        return value_loss

    def compute_entropy_bonus(
        self, logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy bonus to encourage exploration.

        Args:
            logits: Policy logits of shape (batch_size, seq_len, vocab_size)
            attention_mask: Mask for valid tokens of shape (batch_size, seq_len)

        Returns:
            Entropy bonus scalar

        Mathematical operations:
        1. Convert to probabilities: p = softmax(logits)
        2. Compute entropy: entropy = -sum(p * log(p + epsilon), dim=-1)
        3. Mask invalid tokens and average: entropy_bonus = mean(entropy * attention_mask)
        4. Higher entropy = more exploration, lower entropy = more exploitation
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute entropy: H = -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)

        # Apply attention mask and compute mean
        masked_entropy = entropy_per_token * attention_mask.float()
        entropy_bonus = masked_entropy.sum() / attention_mask.float().sum().clamp(min=1)

        return entropy_bonus

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one PPO training step.

        Args:
            batch: Dictionary containing:
                - 'prompts': Prompt token IDs of shape (batch_size, prompt_len)
                - 'responses': Response token IDs of shape (batch_size, response_len)
                - 'old_logits': Old policy logits for responses
                - 'old_values': Old value estimates

        Returns:
            Dictionary of training metrics

        Operations:
        1. Compute current policy logits and values
        2. Compute rewards using reward model
        3. Compute advantages from rewards and values
        4. Compute PPO loss components
        5. Combine losses and update parameters
        6. Return metrics for logging
        """
        prompts = batch["prompts"]
        responses = batch["responses"]
        old_logits = batch["old_logits"]
        old_values = batch["old_values"]

        # Create full sequences and attention masks
        full_sequences = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.ones_like(full_sequences)

        # Only compute gradients for the response part
        response_mask = torch.zeros_like(full_sequences)
        response_mask[:, prompts.size(1) :] = 1

        # Get current policy logits
        current_logits, _ = self.policy_model(
            full_sequences, attention_mask=attention_mask
        )
        # Only take logits for the response part
        response_logits = current_logits[:, prompts.size(1) :, :]

        # Get reference model logits for KL penalty
        with torch.no_grad():
            ref_logits, _ = self.reference_model(
                full_sequences, attention_mask=attention_mask
            )
            ref_response_logits = ref_logits[:, prompts.size(1) :, :]

        # Get current value estimates
        current_values = self.value_model(full_sequences, attention_mask=attention_mask)

        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # Compute advantages
        advantages = self.compute_advantages(rewards, current_values)

        # Compute loss components
        ppo_loss = self.compute_ppo_loss(
            response_logits,
            old_logits,
            responses,
            advantages,
            response_mask[:, prompts.size(1) :],
        )

        value_loss = self.compute_value_loss(current_values, rewards)

        kl_penalty = self.compute_kl_penalty(
            response_logits, ref_response_logits, response_mask[:, prompts.size(1) :]
        )

        entropy_bonus = self.compute_entropy_bonus(
            response_logits, response_mask[:, prompts.size(1) :]
        )

        # Combine losses
        total_loss = (
            ppo_loss
            + self.value_coeff * value_loss
            + self.kl_coeff * kl_penalty
            - self.entropy_coeff * entropy_bonus
        )

        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward(
            retain_graph=True
        )  # Retain graph for value function backward
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update value function separately
        self.value_optimizer.zero_grad()
        value_loss.backward()  # Can free graph now
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        # Return metrics
        return {
            "total_loss": total_loss.item(),
            "ppo_loss": ppo_loss.item(),
            "value_loss": value_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "entropy_bonus": entropy_bonus.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

    def generate_responses(
        self, prompts: torch.Tensor, max_length: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Generate responses to prompts for creating training batches.

        Args:
            prompts: Prompt token IDs of shape (batch_size, prompt_len)
            max_length: Maximum response length

        Returns:
            Dictionary containing generated responses and associated data

        Operations:
        1. Generate responses using current policy
        2. Compute logits for both policy and reference model
        3. Compute value estimates
        4. Store all data needed for PPO training step
        """
        batch_size = prompts.size(0)
        device = prompts.device

        # Generate responses token by token
        responses = []
        current_sequences = prompts.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Get logits from current policy
                logits, _ = self.policy_model(current_sequences)
                next_token_logits = logits[:, -1, :]

                # Sample next token (temperature = 1.0 for exploration)
                next_tokens = torch.multinomial(
                    F.softmax(next_token_logits, dim=-1), num_samples=1
                )

                responses.append(next_tokens)
                current_sequences = torch.cat([current_sequences, next_tokens], dim=1)

        # Stack responses
        responses = torch.cat(responses, dim=1)  # (batch_size, max_length)

        # Create full sequences for computing logits
        full_sequences = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.ones_like(full_sequences)

        # Get policy logits for the response part
        with torch.no_grad():
            policy_logits, _ = self.policy_model(
                full_sequences, attention_mask=attention_mask
            )
            response_logits = policy_logits[:, prompts.size(1) :, :]

            # Get value estimates
            values = self.value_model(full_sequences, attention_mask=attention_mask)

        return {
            "prompts": prompts,
            "responses": responses,
            "old_logits": response_logits,
            "old_values": values,
            "full_sequences": full_sequences,
            "attention_mask": attention_mask,
        }


class RLHFTrainer:
    """Main RLHF training coordinator."""

    def __init__(self, config: GPTConfig, device: str = "cuda"):
        """
        Initialize RLHF trainer with all required models.

        Args:
            config: GPT-2 configuration
            device: Training device
        """
        self.config = config
        self.device = device

        # Initialize models
        self.policy_model = GPT(config).to(device)
        self.reference_model = GPT(config).to(device)
        self.reward_model = RewardModel(config).to(device)
        self.value_model = ValueFunction(config).to(device)

        # Copy policy weights to reference model
        self.reference_model.load_state_dict(self.policy_model.state_dict())

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            self.policy_model, self.value_model, self.reward_model, self.reference_model
        )

    def train_reward_model(
        self, preference_data: List[Tuple[str, str, str]], num_epochs: int = 3
    ) -> None:
        """
        Train reward model on human preference data.

        Args:
            preference_data: List of (prompt, chosen_response, rejected_response) tuples
            num_epochs: Number of training epochs

        Operations:
        1. Convert preference data to token IDs
        2. For each batch: compute pairwise ranking loss
        3. Update reward model parameters
        """
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")

        # Prepare data
        batch_data = []
        for prompt, chosen, rejected in preference_data:
            prompt_ids = tokenizer.encode(prompt)
            chosen_ids = tokenizer.encode(chosen)
            rejected_ids = tokenizer.encode(rejected)

            # Create full sequences
            chosen_seq = prompt_ids + chosen_ids
            rejected_seq = prompt_ids + rejected_ids

            batch_data.append((chosen_seq, rejected_seq))

        # Set up optimizer for reward model
        reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(0, len(batch_data), 4):  # Mini-batch size of 4
                batch = batch_data[i : i + 4]

                # Pad sequences to same length
                chosen_seqs = [torch.tensor(item[0]) for item in batch]
                rejected_seqs = [torch.tensor(item[1]) for item in batch]

                # Pad and stack
                chosen_batch = torch.nn.utils.rnn.pad_sequence(
                    chosen_seqs, batch_first=True, padding_value=0
                ).to(self.device)
                rejected_batch = torch.nn.utils.rnn.pad_sequence(
                    rejected_seqs, batch_first=True, padding_value=0
                ).to(self.device)

                # Get rewards
                chosen_rewards = self.reward_model(chosen_batch)
                rejected_rewards = self.reward_model(rejected_batch)

                # Compute loss
                loss = self.reward_model.compute_pairwise_loss(
                    chosen_rewards, rejected_rewards
                )

                # Update
                reward_optimizer.zero_grad()
                loss.backward()
                reward_optimizer.step()

                total_loss += loss.item()

            print(f"Reward model epoch {epoch+1}/{num_epochs}, loss: {total_loss:.4f}")

    def train_policy_with_ppo(
        self, prompts: List[str], num_epochs: int = 5, batch_size: int = 16
    ) -> None:
        """
        Train policy using PPO with RLHF.

        Args:
            prompts: List of training prompts
            num_epochs: Number of PPO epochs
            batch_size: Training batch size

        Operations:
        1. For each epoch: sample prompts, generate responses, run PPO updates
        2. Log training metrics
        """
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")

        for epoch in range(num_epochs):
            epoch_metrics = []

            # Sample batch of prompts
            import random

            batch_prompts = random.sample(prompts, min(batch_size, len(prompts)))

            # Convert to token IDs
            prompt_ids = []
            for prompt in batch_prompts:
                ids = tokenizer.encode(prompt)
                prompt_ids.append(torch.tensor(ids))

            # Pad and create batch
            prompt_batch = torch.nn.utils.rnn.pad_sequence(
                prompt_ids, batch_first=True, padding_value=0
            ).to(self.device)

            # Generate responses
            batch_data = self.ppo_trainer.generate_responses(
                prompt_batch, max_length=50
            )

            # Perform multiple PPO update steps
            for ppo_step in range(4):  # 4 PPO steps per batch
                metrics = self.ppo_trainer.train_step(batch_data)
                epoch_metrics.append(metrics)

            # Log average metrics for epoch
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(
                    epoch_metrics
                )

            print(f"PPO epoch {epoch+1}/{num_epochs}:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")

    def evaluate_policy(self, test_prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate trained policy on test prompts.

        Args:
            test_prompts: List of evaluation prompts

        Returns:
            Dictionary of evaluation metrics

        Operations:
        1. Generate responses to test prompts
        2. Compute average reward and KL divergence
        """
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")

        total_reward = 0.0
        total_kl = 0.0
        num_samples = 0

        with torch.no_grad():
            for prompt in test_prompts:
                # Convert prompt to tokens
                prompt_ids = (
                    torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
                )

                # Generate response
                batch_data = self.ppo_trainer.generate_responses(
                    prompt_ids, max_length=50
                )

                # Compute reward
                rewards = self.ppo_trainer.compute_rewards(
                    batch_data["prompts"], batch_data["responses"]
                )

                # Compute KL divergence
                policy_logits, _ = self.policy_model(
                    batch_data["full_sequences"],
                    attention_mask=batch_data["attention_mask"],
                )
                ref_logits, _ = self.reference_model(
                    batch_data["full_sequences"],
                    attention_mask=batch_data["attention_mask"],
                )

                response_mask = torch.zeros_like(batch_data["full_sequences"])
                response_mask[:, prompt_ids.size(1) :] = 1

                kl_div = self.ppo_trainer.compute_kl_penalty(
                    policy_logits[:, prompt_ids.size(1) :, :],
                    ref_logits[:, prompt_ids.size(1) :, :],
                    response_mask[:, prompt_ids.size(1) :],
                )

                total_reward += rewards.mean().item()
                total_kl += kl_div.item()
                num_samples += 1

        return {
            "average_reward": total_reward / num_samples,
            "average_kl_divergence": total_kl / num_samples,
        }


def gradient_check(
    model: nn.Module, inputs: Dict[str, torch.Tensor], loss_fn, epsilon: float = 1e-7
) -> bool:
    """
    Numerical gradient checking for model components.

    Args:
        model: Model to check
        inputs: Input tensors
        loss_fn: Loss function that takes model output and returns scalar loss
        epsilon: Finite difference step size

    Returns:
        True if gradients match within tolerance

    Mathematical operations:
    1. For each parameter: compute analytical and numerical gradients
    2. Check relative error between them
    3. Return True if all parameters pass
    """
    model.eval()
    tolerance = 1e-3

    # Get analytical gradients
    model.zero_grad()
    output = model(**inputs)
    loss = loss_fn(output)
    loss.backward()

    passed = True

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Skip embedding layers as they have sparse gradients
        if "wte.weight" in name or "wpe.weight" in name:
            continue

        # Skip LayerNorm layers as they have numerical sensitivity
        if "ln_1." in name or "ln_2." in name or "ln_f." in name:
            continue

        # Sample a few random elements to check (for efficiency)
        flat_param = param.view(-1)
        flat_grad = param.grad.view(-1)

        # Check up to 10 random elements
        num_checks = min(10, flat_param.numel())
        indices = torch.randperm(flat_param.numel())[:num_checks]

        for idx in indices:
            # Save original value
            original_value = flat_param[idx].item()

            # Compute f(θ + ε)
            with torch.no_grad():
                flat_param[idx] = original_value + epsilon
            model.zero_grad()
            output_plus = model(**inputs)
            loss_plus = loss_fn(output_plus)

            # Compute f(θ - ε)
            with torch.no_grad():
                flat_param[idx] = original_value - epsilon
            model.zero_grad()
            output_minus = model(**inputs)
            loss_minus = loss_fn(output_minus)

            # Restore original value
            with torch.no_grad():
                flat_param[idx] = original_value

            # Compute numerical gradient
            numerical_grad = (loss_plus.item() - loss_minus.item()) / (2 * epsilon)
            analytical_grad = flat_grad[idx].item()

            # Compute relative error
            if abs(analytical_grad) > 1e-8 or abs(numerical_grad) > 1e-8:
                relative_error = abs(analytical_grad - numerical_grad) / max(
                    abs(analytical_grad), abs(numerical_grad), 1e-8
                )

                if relative_error > tolerance:
                    print(
                        f"Gradient check failed for {name}[{idx}]: "
                        f"analytical={analytical_grad:.6f}, "
                        f"numerical={numerical_grad:.6f}, "
                        f"error={relative_error:.6f}"
                    )
                    passed = False

    return passed


def main():
    """
    Main training script demonstrating RLHF PPO usage.
    """
    # Initialize configuration and models
    config = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create trainer
    trainer = RLHFTrainer(config, device)

    # Example preference data (prompt, chosen, rejected)
    preference_data = [
        (
            "Explain photosynthesis",
            "Photosynthesis is the process...",
            "Plants eat sunlight...",
        ),
        (
            "How do computers work?",
            "Computers process information...",
            "Magic happens inside...",
        ),
    ]

    # Example training prompts
    training_prompts = [
        "Explain quantum mechanics",
        "Write a short story about space",
        "How does machine learning work?",
    ]

    # Train reward model
    print("Training reward model...")
    trainer.train_reward_model(preference_data)

    # Train policy with PPO
    print("Training policy with PPO...")
    trainer.train_policy_with_ppo(training_prompts)

    # Evaluate final policy
    print("Evaluating policy...")
    results = trainer.evaluate_policy(training_prompts[:2])
    print(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
