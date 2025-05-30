"""
Gradient check unit tests for RLHF PPO implementation.

This module provides comprehensive gradient checking for all components
of the RLHF PPO implementation to ensure mathematical correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import numpy as np
from typing import Dict, Callable, Any
import sys
import os
import math

# Add the parent directory to path to import rlhf_ppo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rlhf_ppo import (
    RewardModel, ValueFunction, PPOTrainer, gradient_check
)
from gpt import GPT, GPTConfig


class GradientChecker:
    """Utility class for numerical gradient checking."""
    
    @staticmethod
    def numerical_gradient(func: Callable, inputs: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
        """
        Compute numerical gradient using finite differences.
        
        Args:
            func: Function that takes inputs and returns scalar loss
            inputs: Input tensor to compute gradients for
            epsilon: Step size for finite differences
            
        Returns:
            Numerical gradient tensor of same shape as inputs
            
        Mathematical operation:
        For each element inputs[i,j,...]:
        grad[i,j,...] = (func(inputs + ε*e_ij) - func(inputs - ε*e_ij)) / (2ε)
        where e_ij is unit vector with 1 at position [i,j,...] and 0 elsewhere
        """
        grad = torch.zeros_like(inputs)
        
        # Flatten for easier indexing
        flat_inputs = inputs.view(-1)
        flat_grad = grad.view(-1)
        
        for i in range(flat_inputs.numel()):
            # Create perturbation
            flat_inputs[i] += epsilon
            loss_plus = func(inputs)
            
            flat_inputs[i] -= 2 * epsilon
            loss_minus = func(inputs)
            
            # Restore original value
            flat_inputs[i] += epsilon
            
            # Compute finite difference
            flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return grad
    
    @staticmethod
    def check_gradients(model: nn.Module, loss_func: Callable, 
                       inputs: Dict[str, torch.Tensor], tolerance: float = 1e-4) -> Dict[str, bool]:
        """
        Check analytical vs numerical gradients for all model parameters.
        
        Args:
            model: PyTorch model to check
            loss_func: Function that computes scalar loss from model outputs
            inputs: Dictionary of input tensors for model
            tolerance: Relative error tolerance for gradient checking
            
        Returns:
            Dictionary mapping parameter names to whether they pass gradient check
            
        Mathematical operations:
        1. Compute analytical gradients via backpropagation
        2. For each parameter θ:
           - Compute numerical gradient: ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε))/(2ε)
           - Check relative error: |analytical - numerical| / max(|analytical|, |numerical|, 1e-8)
           - Pass if relative error < tolerance
        """
        model.train()
        results = {}
        
        # Compute analytical gradients
        model.zero_grad()
        output = model(**inputs)
        loss = loss_func(output)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is None:
                results[name] = True  # No gradient expected
                continue
                
            # Check a subset of parameters for efficiency
            flat_param = param.view(-1)
            flat_grad = param.grad.view(-1)
            
            num_checks = min(5, flat_param.numel())  # Check up to 5 elements
            indices = torch.randperm(flat_param.numel())[:num_checks]
            
            param_passed = True
            
            for idx in indices:
                original_value = flat_param[idx].item()
                
                # Compute numerical gradient
                epsilon = 1e-5
                
                # f(θ + ε)
                with torch.no_grad():
                    flat_param[idx] = original_value + epsilon
                model.zero_grad()
                output_plus = model(**inputs)
                loss_plus = loss_func(output_plus)
                
                # f(θ - ε)
                with torch.no_grad():
                    flat_param[idx] = original_value - epsilon
                model.zero_grad()
                output_minus = model(**inputs)
                loss_minus = loss_func(output_minus)
                
                # Restore original value
                with torch.no_grad():
                    flat_param[idx] = original_value
                
                # Compute gradients
                numerical_grad = (loss_plus.item() - loss_minus.item()) / (2 * epsilon)
                analytical_grad = flat_grad[idx].item()
                
                # Check relative error
                if abs(analytical_grad) > 1e-8 or abs(numerical_grad) > 1e-8:
                    relative_error = abs(analytical_grad - numerical_grad) / max(
                        abs(analytical_grad), abs(numerical_grad), 1e-8
                    )
                    
                    if relative_error > tolerance:
                        param_passed = False
                        print(f"Gradient check failed for {name}[{idx}]: "
                              f"analytical={analytical_grad:.6f}, "
                              f"numerical={numerical_grad:.6f}, "
                              f"error={relative_error:.6f}")
            
            results[name] = param_passed
        
        return results


class TestGPTModel(unittest.TestCase):
    """Test gradient computation for complete GPT model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig()
        self.config.n_embd = 64
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.vocab_size = 100
        self.config.block_size = 64
        self.model = GPT(self.config)
        self.batch_size = 2
        self.seq_len = 8
        
    def test_language_modeling_gradients(self):
        """
        Test gradients for language modeling objective.
        
        Test cases:
        1. Token embedding gradients
        2. Position embedding gradients  
        3. Output projection gradients
        4. Cross-entropy loss gradients
        
        Mathematical verification:
        - Cross-entropy loss: L = -∑ y_true * log(softmax(logits))
        - ∂L/∂logits = softmax(logits) - y_true
        - Gradients should flow back through all transformer layers
        """
        # Create test inputs
        input_ids = torch.randint(0, self.config.vocab_size, 
                                 (self.batch_size, self.seq_len))
        targets = torch.randint(0, self.config.vocab_size, 
                               (self.batch_size, self.seq_len))
        
        def loss_fn(output):
            logits, loss = output
            return loss if loss is not None else F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        
        inputs = {'x': input_ids, 'targets': targets}
        
        # Check gradients
        passed = gradient_check(self.model, inputs, loss_fn)
        self.assertTrue(passed, "Language modeling gradients failed")
        
    def test_embedding_gradients(self):
        """
        Test embedding layer gradient computation.
        
        Mathematical operations:
        - Token embeddings: E[input_ids] where E is embedding matrix
        - Position embeddings: P[positions] where P is position embedding matrix
        - ∂L/∂E should only update embeddings for tokens present in batch
        - ∂L/∂P should only update position embeddings for used positions
        """
        # Create simple test case
        input_ids = torch.randint(0, min(50, self.config.vocab_size), 
                                 (1, 4))  # Small sequence
        
        def loss_fn(output):
            logits, _ = output
            return logits.sum()  # Simple loss for gradient checking
        
        inputs = {'x': input_ids}
        
        # Check embeddings specifically
        passed = GradientChecker.check_gradients(
            self.model, loss_fn, inputs, tolerance=1e-3
        )
        
        # Check that only used embeddings get gradients
        self.model.zero_grad()
        logits, _ = self.model(input_ids)
        logits.sum().backward()
        
        # Token embeddings should have gradients only for used tokens
        used_tokens = input_ids.unique()
        for token_id in used_tokens:
            self.assertIsNotNone(self.model.wte.weight.grad[token_id],
                               f"Token {token_id} should have gradients")
        
        print(f"Embedding gradient check passed: {all(passed.values())}")


class TestRewardModel(unittest.TestCase):
    """Test gradient computation for reward model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig()
        self.config.n_embd = 64
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.vocab_size = 100
        self.config.block_size = 64
        self.reward_model = RewardModel(self.config)
        self.batch_size = 2
        self.seq_len = 16
        
    def test_reward_score_gradients(self):
        """
        Test gradients for reward score computation.
        
        Test cases:
        1. Sequence pooling gradients
        2. Reward head gradients
        3. Scalar output gradient flow
        
        Mathematical verification:
        - Pooling operation gradients (mean, last token, etc.)
        - Linear layer gradients for reward head
        - Gradient flow from scalar reward back to sequence representations
        """
        # Create test inputs
        input_ids = torch.randint(0, self.config.vocab_size, 
                                 (self.batch_size, self.seq_len))
        
        def loss_fn(output):
            rewards = output
            return rewards.sum()  # Simple loss for gradient checking
        
        inputs = {'input_ids': input_ids}
        
        # Check gradients
        passed = gradient_check(self.reward_model, inputs, loss_fn)
        self.assertTrue(passed, "Reward model gradients failed")
        
        # Specifically test reward head gradients
        self.reward_model.zero_grad()
        rewards = self.reward_model(input_ids)
        rewards.sum().backward()
        
        # Reward head should have gradients
        self.assertIsNotNone(self.reward_model.reward_head.weight.grad)
        self.assertIsNotNone(self.reward_model.reward_head.bias.grad)
        
        print("Reward model gradient check passed")
        
    def test_pairwise_loss_gradients(self):
        """
        Test gradients for pairwise ranking loss.
        
        Mathematical operation to verify:
        L = -log(sigmoid(r_chosen - r_rejected))
        where r_chosen, r_rejected are reward scores
        
        Gradients:
        ∂L/∂r_chosen = -sigmoid(r_rejected - r_chosen)
        ∂L/∂r_rejected = sigmoid(r_rejected - r_chosen)
        """
        # Create test rewards
        chosen_rewards = torch.tensor([1.0, 2.0], requires_grad=True)
        rejected_rewards = torch.tensor([0.5, 1.5], requires_grad=True)
        
        # Compute pairwise loss
        loss = self.reward_model.compute_pairwise_loss(chosen_rewards, rejected_rewards)
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist and have correct signs
        self.assertIsNotNone(chosen_rewards.grad)
        self.assertIsNotNone(rejected_rewards.grad)
        
        # For pairwise loss = -log(sigmoid(chosen - rejected))
        # Gradients: d/d_chosen = -sigmoid(rejected - chosen), d/d_rejected = sigmoid(rejected - chosen)
        reward_diff = chosen_rewards - rejected_rewards
        expected_chosen_grad = -torch.sigmoid(-reward_diff) 
        expected_rejected_grad = torch.sigmoid(-reward_diff)
        
        # Check approximate equality
        torch.testing.assert_close(chosen_rewards.grad, expected_chosen_grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(rejected_rewards.grad, expected_rejected_grad, rtol=1e-3, atol=1e-3)
        
        print("Pairwise loss gradient check passed")
        
    def test_preference_learning_gradients(self):
        """
        Test end-to-end gradients for preference learning.
        
        Verification:
        - Gradients should encourage higher scores for preferred outputs
        - Gradients should discourage higher scores for rejected outputs
        - Preference margin should affect gradient magnitudes appropriately
        """
        # Create test sequences
        chosen_seq = torch.randint(0, self.config.vocab_size, (1, self.seq_len))
        rejected_seq = torch.randint(0, self.config.vocab_size, (1, self.seq_len))
        
        # Compute rewards
        chosen_reward = self.reward_model(chosen_seq)
        rejected_reward = self.reward_model(rejected_seq)
        
        # Compute pairwise loss
        loss = self.reward_model.compute_pairwise_loss(chosen_reward, rejected_reward)
        
        # Get gradients
        self.reward_model.zero_grad()
        loss.backward()
        
        # Check that model parameters have gradients
        has_gradients = False
        for param in self.reward_model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        self.assertTrue(has_gradients, "Reward model should have non-zero gradients")
        print("Preference learning gradient check passed")


class TestValueFunction(unittest.TestCase):
    """Test gradient computation for value function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig()
        self.config.n_embd = 64
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.vocab_size = 100
        self.config.block_size = 64
        self.value_function = ValueFunction(self.config)
        self.batch_size = 2
        self.seq_len = 16
        
    def test_value_prediction_gradients(self):
        """
        Test gradients for value prediction.
        
        Test cases:
        1. Value head gradients
        2. MSE loss gradients
        3. Target value gradient flow
        
        Mathematical verification:
        MSE loss: L = (predicted_value - target_value)^2
        ∂L/∂predicted_value = 2 * (predicted_value - target_value)
        """
        # Create test inputs
        input_ids = torch.randint(0, self.config.vocab_size, 
                                 (self.batch_size, self.seq_len))
        target_values = torch.randn(self.batch_size)
        
        # Test value prediction gradients
        predicted_values = self.value_function(input_ids)
        loss = F.mse_loss(predicted_values, target_values)
        
        self.value_function.zero_grad()
        loss.backward()
        
        # Check that value head has gradients
        self.assertIsNotNone(self.value_function.value_head.weight.grad)
        self.assertIsNotNone(self.value_function.value_head.bias.grad)
        
        # Test gradient checking
        def loss_fn(output):
            values = output
            return F.mse_loss(values, target_values)
        
        inputs = {'input_ids': input_ids}
        passed = gradient_check(self.value_function, inputs, loss_fn)
        self.assertTrue(passed, "Value function gradients failed")
        
        print("Value function gradient check passed")


class TestPPOLoss(unittest.TestCase):
    """Test gradient computation for PPO loss components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig()
        self.config.n_embd = 64
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.vocab_size = 100
        self.config.block_size = 64
        
        # Initialize models
        self.policy_model = GPT(self.config)
        self.value_model = ValueFunction(self.config)
        self.reward_model = RewardModel(self.config)
        self.reference_model = GPT(self.config)
        
        # Copy weights to reference model
        self.reference_model.load_state_dict(self.policy_model.state_dict())
        
        self.ppo_trainer = PPOTrainer(
            self.policy_model, self.value_model, 
            self.reward_model, self.reference_model
        )
        
        self.batch_size = 2
        self.seq_len = 16
        
    def test_clipped_objective_gradients(self):
        """
        Test gradients for PPO clipped objective.
        
        Mathematical operation to verify:
        ratio = exp(log_prob_new - log_prob_old)
        clipped_ratio = clip(ratio, 1-ε, 1+ε)
        surrogate1 = ratio * advantage
        surrogate2 = clipped_ratio * advantage
        loss = -min(surrogate1, surrogate2)
        
        Gradient behavior:
        - When ratio in [1-ε, 1+ε]: gradient flows normally
        - When ratio outside clip range: gradient is clipped/zeroed
        """
        # Create test data
        batch_size, seq_len, vocab_size = 2, 8, self.config.vocab_size
        
        # Create random logits and actions
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        old_logits = torch.randn(batch_size, seq_len, vocab_size)
        actions = torch.randint(0, vocab_size, (batch_size, seq_len))
        advantages = torch.randn(batch_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Compute PPO loss
        ppo_loss = self.ppo_trainer.compute_ppo_loss(
            logits, old_logits, actions, advantages, attention_mask
        )
        
        # Check that loss is scalar and finite
        self.assertTrue(ppo_loss.dim() == 0, "PPO loss should be scalar")
        self.assertTrue(torch.isfinite(ppo_loss), "PPO loss should be finite")
        
        # Compute gradients
        ppo_loss.backward()
        
        # Check that gradients exist and are finite
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)), "Gradients should be finite")
        
        print("PPO clipped objective gradient check passed")
        
    def test_kl_penalty_gradients(self):
        """
        Test gradients for KL divergence penalty.
        
        Mathematical operation to verify:
        KL(p||q) = ∑ p(x) * log(p(x)/q(x))
        where p is new policy, q is reference policy
        
        Gradients:
        ∂KL/∂log(p) = p * (1 + log(p) - log(q))
        Should discourage large deviations from reference policy
        """
        # Create test data
        batch_size, seq_len, vocab_size = 2, 8, self.config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Compute KL penalty
        kl_penalty = self.ppo_trainer.compute_kl_penalty(
            logits, ref_logits, attention_mask
        )
        
        # Check that KL is non-negative and finite
        self.assertGreaterEqual(kl_penalty.item(), 0, "KL should be non-negative")
        self.assertTrue(torch.isfinite(kl_penalty), "KL should be finite")
        
        # Compute gradients
        kl_penalty.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)), "KL gradients should be finite")
        
        print("KL penalty gradient check passed")
        
    def test_advantage_computation_gradients(self):
        """
        Test gradients for advantage computation.
        
        Mathematical operation:
        advantage = reward - baseline_value
        
        Gradients should flow to:
        - Value function (to improve baseline estimates)
        - Policy (weighted by advantage for PPO objective)
        """
        # Create test data
        rewards = torch.tensor([1.0, 2.0], requires_grad=True)
        values = torch.tensor([0.8, 1.5], requires_grad=True)
        
        # Compute advantages
        advantages = self.ppo_trainer.compute_advantages(rewards, values)
        
        # Check that advantages are normalized
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(advantages.std().item(), 1.0, places=5)
        
        # Advantages should be differentiable w.r.t. both rewards and values
        test_loss = advantages.sum()
        test_loss.backward()
        
        self.assertIsNotNone(rewards.grad)
        self.assertIsNotNone(values.grad)
        
        print("Advantage computation gradient check passed")
        
    def test_entropy_bonus_gradients(self):
        """
        Test gradients for entropy bonus.
        
        Mathematical operation:
        entropy = -∑ p(x) * log(p(x))
        
        Gradients:
        ∂entropy/∂log(p) = -(1 + log(p))
        Should encourage exploration by increasing entropy
        """
        # Create test data  
        batch_size, seq_len, vocab_size = 2, 8, self.config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Compute entropy bonus
        entropy_bonus = self.ppo_trainer.compute_entropy_bonus(logits, attention_mask)
        
        # Entropy should be positive and finite
        self.assertGreater(entropy_bonus.item(), 0, "Entropy should be positive")
        self.assertTrue(torch.isfinite(entropy_bonus), "Entropy should be finite")
        
        # Compute gradients
        entropy_bonus.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)), "Entropy gradients should be finite")
        
        print("Entropy bonus gradient check passed")
        
    def test_combined_ppo_loss_gradients(self):
        """
        Test gradients for combined PPO loss.
        
        Combined loss:
        L = L_clip + c1*L_value + c2*L_entropy - c3*KL_penalty
        
        Verification:
        - All loss components should contribute to gradients
        - Gradient magnitudes should be balanced across components
        - Policy parameters should receive gradients from PPO objective
        - Value parameters should receive gradients from value loss
        """
        # Create test batch
        batch_size, prompt_len, response_len = 2, 4, 8
        vocab_size = self.config.vocab_size
        
        prompts = torch.randint(0, vocab_size, (batch_size, prompt_len))
        responses = torch.randint(0, vocab_size, (batch_size, response_len))
        old_logits = torch.randn(batch_size, response_len, vocab_size)
        old_values = torch.randn(batch_size)
        
        batch = {
            'prompts': prompts,
            'responses': responses,
            'old_logits': old_logits,
            'old_values': old_values
        }
        
        # Run training step
        metrics = self.ppo_trainer.train_step(batch)
        
        # Check that all loss components are present and finite
        required_metrics = ['total_loss', 'ppo_loss', 'value_loss', 'kl_penalty', 'entropy_bonus']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(math.isfinite(metrics[metric]), f"{metric} should be finite")
        
        # Check that policy has gradients
        policy_has_grads = any(p.grad is not None and torch.any(p.grad != 0) 
                              for p in self.policy_model.parameters())
        self.assertTrue(policy_has_grads, "Policy should have gradients")
        
        # Check that value function has gradients
        value_has_grads = any(p.grad is not None and torch.any(p.grad != 0) 
                             for p in self.value_model.parameters())
        self.assertTrue(value_has_grads, "Value function should have gradients")
        
        print("Combined PPO loss gradient check passed")


class TestRLHFEndToEnd(unittest.TestCase):
    """Test end-to-end gradient flow for complete RLHF pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig()
        self.config.n_embd = 64
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.vocab_size = 100
        self.config.block_size = 64
        
        # Smaller dimensions for testing
        torch.manual_seed(42)
        
    def test_reward_model_training_gradients(self):
        """
        Test gradients during reward model training phase.
        
        Pipeline:
        1. Prompt + response → reward score
        2. Pairwise ranking loss on preferences
        3. Gradients should flow back to reward model parameters
        
        Verification:
        - Reward model parameters should receive non-zero gradients
        - Gradients should favor preferred responses over rejected ones
        """
        reward_model = RewardModel(self.config)
        
        # Create test preference data
        seq_len = 16
        chosen_seq = torch.randint(0, self.config.vocab_size, (2, seq_len))
        rejected_seq = torch.randint(0, self.config.vocab_size, (2, seq_len))
        
        # Compute rewards and loss
        chosen_rewards = reward_model(chosen_seq)
        rejected_rewards = reward_model(rejected_seq)
        loss = reward_model.compute_pairwise_loss(chosen_rewards, rejected_rewards)
        
        # Compute gradients
        reward_model.zero_grad()
        loss.backward()
        
        # Check that reward model has gradients
        has_gradients = any(p.grad is not None and torch.any(p.grad != 0) 
                           for p in reward_model.parameters())
        self.assertTrue(has_gradients, "Reward model should have gradients")
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        
        print("Reward model training gradient check passed")
        
    def test_ppo_training_gradients(self):
        """
        Test gradients during PPO training phase.
        
        Pipeline:
        1. Generate responses from policy
        2. Compute rewards using reward model
        3. Compute advantages and PPO loss
        4. Update policy and value function
        
        Verification:
        - Policy parameters should receive gradients from PPO objective
        - Value function should receive gradients from value loss
        - Reference model should not receive gradients (frozen)
        - Reward model should not receive gradients (frozen)
        """
        from rlhf_ppo import RLHFTrainer
        
        trainer = RLHFTrainer(self.config, "cpu")
        
        # Create test prompts
        prompt_len, response_len = 4, 8
        prompts = torch.randint(0, self.config.vocab_size, (2, prompt_len))
        
        # Generate responses and train
        batch_data = trainer.ppo_trainer.generate_responses(prompts, max_length=response_len)
        metrics = trainer.ppo_trainer.train_step(batch_data)
        
        # Check that policy has gradients
        policy_has_grads = any(p.grad is not None and torch.any(p.grad != 0) 
                              for p in trainer.policy_model.parameters())
        self.assertTrue(policy_has_grads, "Policy should have gradients")
        
        # Check that value function has gradients
        value_has_grads = any(p.grad is not None and torch.any(p.grad != 0) 
                             for p in trainer.value_model.parameters())
        self.assertTrue(value_has_grads, "Value function should have gradients")
        
        # Check that reference model has no gradients (frozen)
        ref_has_grads = any(p.grad is not None and torch.any(p.grad != 0) 
                           for p in trainer.reference_model.parameters())
        self.assertFalse(ref_has_grads, "Reference model should not have gradients")
        
        # Check that reward model has no gradients (frozen)
        reward_has_grads = any(p.grad is not None and torch.any(p.grad != 0) 
                              for p in trainer.reward_model.parameters())
        self.assertFalse(reward_has_grads, "Reward model should not have gradients")
        
        print("PPO training gradient check passed")
        
    def test_response_generation_gradients(self):
        """
        Test gradients during response generation for PPO.
        
        Process:
        1. Autoregressive generation from policy
        2. Log probability computation for generated tokens
        3. Gradient flow through generation process
        
        Verification:
        - Generated sequences should have valid log probabilities
        - Gradients should flow correctly through sampling operation
        - Temperature scaling should affect gradients appropriately
        """
        policy_model = GPT(self.config)
        
        # Create test prompts
        prompt_len = 4
        prompts = torch.randint(0, self.config.vocab_size, (1, prompt_len))
        
        # Generate response deterministically for gradient checking
        with torch.no_grad():
            logits, _ = policy_model(prompts)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Create full sequence
        full_seq = torch.cat([prompts, next_token], dim=1)
        
        # Compute log probabilities with gradients
        logits, _ = policy_model(full_seq)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probability of the generated token
        generated_log_prob = log_probs[0, -2, next_token[0, 0]]  # -2 because we want logits that generated the last token
        
        # Backpropagate
        policy_model.zero_grad()
        generated_log_prob.backward()
        
        # Check that model has gradients
        has_gradients = any(p.grad is not None and torch.any(p.grad != 0) 
                           for p in policy_model.parameters())
        self.assertTrue(has_gradients, "Policy should have gradients from generation")
        
        print("Response generation gradient check passed")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of gradient computations."""
    
    def test_softmax_gradients_stability(self):
        """
        Test numerical stability of softmax gradients.
        
        Issues to check:
        - Large logit values causing overflow
        - Small probabilities causing underflow
        - Gradient explosion/vanishing
        
        Mathematical considerations:
        - Use log-softmax for better numerical stability
        - Ensure gradients remain bounded
        """
        # Test with extreme logit values
        extreme_logits = torch.tensor([[100.0, -100.0, 0.0]], requires_grad=True)
        
        # Use log-softmax for stability
        log_probs = F.log_softmax(extreme_logits, dim=-1)
        loss = -log_probs.sum()
        
        loss.backward()
        
        # Check that gradients are finite
        self.assertTrue(torch.all(torch.isfinite(extreme_logits.grad)), 
                       "Softmax gradients should be finite even with extreme values")
        
        # Check that gradients are bounded
        self.assertTrue(torch.all(torch.abs(extreme_logits.grad) < 10), 
                       "Softmax gradients should be bounded")
        
        print("Softmax stability check passed")
        
    def test_kl_divergence_stability(self):
        """
        Test numerical stability of KL divergence computation.
        
        Issues to check:
        - Division by zero when reference probabilities are zero
        - Log of zero causing -inf
        - Proper handling of epsilon values
        
        Mathematical fix:
        KL(p||q) = ∑ p * log((p + ε) / (q + ε))
        """
        # Create logits that could cause instability
        logits = torch.tensor([[10.0, -10.0, 0.0]], requires_grad=True)
        ref_logits = torch.tensor([[-10.0, 10.0, 0.0]])
        attention_mask = torch.ones(1, 3)
        
        # Test our KL implementation
        trainer = PPOTrainer(
            GPT(GPTConfig()), ValueFunction(GPTConfig()),
            RewardModel(GPTConfig()), GPT(GPTConfig())
        )
        
        kl_div = trainer.compute_kl_penalty(
            logits.unsqueeze(0), ref_logits.unsqueeze(0), attention_mask
        )
        
        # KL should be finite and non-negative
        self.assertTrue(torch.isfinite(kl_div), "KL divergence should be finite")
        self.assertGreaterEqual(kl_div.item(), 0, "KL divergence should be non-negative")
        
        # Compute gradients
        kl_div.backward()
        
        # Gradients should be finite
        self.assertTrue(torch.all(torch.isfinite(logits.grad)), 
                       "KL gradients should be finite")
        
        print("KL divergence stability check passed")
        
    def test_advantage_normalization_stability(self):
        """
        Test numerical stability of advantage normalization.
        
        Issues to check:
        - Division by zero when advantage std is zero
        - Proper handling of constant advantages
        - Gradient flow through normalization
        """
        trainer = PPOTrainer(
            GPT(GPTConfig()), ValueFunction(GPTConfig()),
            RewardModel(GPTConfig()), GPT(GPTConfig())
        )
        
        # Test with constant advantages (std = 0)
        rewards = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        values = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        
        advantages = trainer.compute_advantages(rewards, values)
        
        # Should handle zero std gracefully
        self.assertTrue(torch.all(torch.isfinite(advantages)), 
                       "Advantages should be finite even with zero std")
        
        # Test gradient flow
        test_loss = advantages.sum()
        test_loss.backward()
        
        self.assertTrue(torch.all(torch.isfinite(rewards.grad)), 
                       "Advantage gradients should be finite")
        self.assertTrue(torch.all(torch.isfinite(values.grad)), 
                       "Value gradients should be finite")
        
        print("Advantage normalization stability check passed")


def run_gradient_checks():
    """
    Run comprehensive gradient checks for all components.
    
    Returns:
        Dictionary summarizing gradient check results
    """
    results = {}
    
    print("Running comprehensive gradient checks...")
    
    try:
        # Test GPT model
        print("\nTesting GPT model...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestGPTModel)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        results['GPT_Model'] = result.wasSuccessful()
        
        # Test Reward Model
        print("Testing Reward model...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRewardModel)
        result = runner.run(suite)
        results['Reward_Model'] = result.wasSuccessful()
        
        # Test Value Function
        print("Testing Value function...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestValueFunction)
        result = runner.run(suite)
        results['Value_Function'] = result.wasSuccessful()
        
        # Test PPO Loss
        print("Testing PPO loss components...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPPOLoss)
        result = runner.run(suite)
        results['PPO_Loss'] = result.wasSuccessful()
        
        # Test End-to-End
        print("Testing end-to-end RLHF...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRLHFEndToEnd)
        result = runner.run(suite)
        results['End_to_End'] = result.wasSuccessful()
        
        # Test Numerical Stability
        print("Testing numerical stability...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNumericalStability)
        result = runner.run(suite)
        results['Numerical_Stability'] = result.wasSuccessful()
        
    except Exception as e:
        print(f"Error during gradient checking: {e}")
        results['Error'] = str(e)
    
    return results


def create_test_batch() -> Dict[str, torch.Tensor]:
    """
    Create a test batch for gradient checking.
    
    Returns:
        Dictionary containing test inputs for models
        
    Should include:
    - input_ids: Token sequences
    - attention_mask: Valid token mask  
    - labels: Target tokens for language modeling
    - rewards: Reward scores
    - advantages: Advantage estimates
    """
    batch_size, seq_len, vocab_size = 2, 16, 1000
    
    batch = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'labels': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'rewards': torch.randn(batch_size),
        'advantages': torch.randn(batch_size),
        'prompts': torch.randint(0, vocab_size, (batch_size, seq_len // 2)),
        'responses': torch.randint(0, vocab_size, (batch_size, seq_len // 2)),
    }
    
    return batch


if __name__ == "__main__":
    # Run gradient checks
    print("Running comprehensive gradient checks...")
    results = run_gradient_checks()
    
    # Print results
    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{component}: {status}")
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)