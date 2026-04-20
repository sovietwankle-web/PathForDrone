"""
Phase 2: PPO Reinforcement Learning Trainer.

Fine-tunes the pre-trained attention network using PPO
to minimize makespan through autoregressive decoding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..config import NeuralAllocatorConfig, TrainingConfig
from ..model import WaypointAllocationNet
from .reward import compute_makespan_reward, compute_step_reward, compute_robot_costs


@dataclass
class RolloutStep:
    """One step in an episode."""
    robot_feats: torch.Tensor       # (n_r, 7)
    waypoint_feats: torch.Tensor    # (n_w, 5)
    pairwise_feats: torch.Tensor    # (n_r, n_w, 2)
    assigned_mask: torch.Tensor     # (n_w,) bool
    robot_mask: torch.Tensor        # (n_r,) bool
    waypoint_mask: torch.Tensor     # (n_w,) bool
    action: int                     # flattened index into (n_r * n_w)
    log_prob: torch.Tensor          # scalar
    reward: float
    value: torch.Tensor             # scalar


class PPOTrainer:
    """PPO fine-tuning trainer with autoregressive rollouts."""

    def __init__(self,
                 model: WaypointAllocationNet,
                 config: TrainingConfig,
                 model_config: Optional[NeuralAllocatorConfig] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.mcfg = model_config or model.config
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.mcfg.lr_rl,
        )

    def rollout_episode(self,
                        robot_feats: np.ndarray,
                        waypoint_feats: np.ndarray,
                        pairwise_feats: np.ndarray,
                        cost_matrix: np.ndarray,
                        wp_to_wp_cost: np.ndarray,
                        n_robots: int,
                        n_waypoints: int
                        ) -> Tuple[List[RolloutStep], float]:
        """
        Run one episode with autoregressive decoding.

        Returns list of RolloutSteps and episode makespan reward.
        """
        self.model.eval()

        # Convert to tensors
        r_feats = torch.from_numpy(robot_feats).unsqueeze(0).to(self.device)
        w_feats = torch.from_numpy(waypoint_feats).unsqueeze(0).to(self.device)
        p_feats = torch.from_numpy(pairwise_feats).unsqueeze(0).to(self.device)

        r_mask = torch.zeros(1, r_feats.shape[1], dtype=torch.bool, device=self.device)
        r_mask[0, :n_robots] = True
        w_mask = torch.zeros(1, w_feats.shape[1], dtype=torch.bool, device=self.device)
        w_mask[0, :n_waypoints] = True

        assigned_mask = torch.zeros(1, w_feats.shape[1], dtype=torch.bool, device=self.device)

        steps = []
        assignments = {r: [] for r in range(n_robots)}
        prev_max_cost = 0.0

        for step_idx in range(n_waypoints):
            with torch.no_grad():
                scores, value = self.model.forward_rl(
                    r_feats, w_feats, p_feats,
                    assigned_mask, r_mask, w_mask
                )

            # Sample action from (n_r, n_w) distribution
            s = scores[0, :n_robots, :n_waypoints]
            flat_logits = s.reshape(-1)

            # Mask out already-assigned
            valid = ~torch.all(torch.isinf(flat_logits) & (flat_logits < 0))
            if not valid:
                break

            probs = torch.softmax(flat_logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            r_id = int(action) // n_waypoints
            w_id = int(action) % n_waypoints

            # Store the assigned_mask BEFORE this assignment (for re-evaluation)
            pre_assign_mask = assigned_mask[0].clone().cpu()

            # Update assignments
            assignments[r_id].append(w_id)
            assigned_mask[0, w_id] = True

            # Compute step reward (dense)
            reward, prev_max_cost = compute_step_reward(
                prev_max_cost, assignments, cost_matrix, wp_to_wp_cost
            )

            steps.append(RolloutStep(
                robot_feats=r_feats.squeeze(0).cpu(),
                waypoint_feats=w_feats.squeeze(0).cpu(),
                pairwise_feats=p_feats.squeeze(0).cpu(),
                assigned_mask=pre_assign_mask,
                robot_mask=r_mask[0].cpu(),
                waypoint_mask=w_mask[0].cpu(),
                action=int(action),
                log_prob=log_prob.cpu(),
                reward=reward,
                value=value[0].cpu(),
            ))

        episode_reward = compute_makespan_reward(assignments, cost_matrix, wp_to_wp_cost)
        return steps, episode_reward

    def compute_gae(self, steps: List[RolloutStep],
                    gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        n = len(steps)
        if n == 0:
            return torch.tensor([]), torch.tensor([])

        rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32)
        values = torch.tensor([s.value.item() for s in steps], dtype=torch.float32)

        advantages = torch.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(self, all_steps: List[RolloutStep],
                   all_advantages: torch.Tensor,
                   all_returns: torch.Tensor) -> Dict[str, float]:
        """Run one PPO update epoch over collected rollout data."""
        self.model.train()

        n = len(all_steps)
        if n == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Shuffle indices
        indices = np.random.permutation(n)
        minibatch_size = max(1, n // self.config.n_minibatches)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            mb_idx = indices[start:end]

            policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            value_loss = torch.tensor(0.0, device=self.device)
            entropy_sum = torch.tensor(0.0, device=self.device)

            for idx in mb_idx:
                step = all_steps[idx]

                # Re-evaluate
                r_f = step.robot_feats.unsqueeze(0).to(self.device)
                w_f = step.waypoint_feats.unsqueeze(0).to(self.device)
                p_f = step.pairwise_feats.unsqueeze(0).to(self.device)
                a_m = step.assigned_mask.unsqueeze(0).to(self.device)
                r_m = step.robot_mask.unsqueeze(0).to(self.device)
                w_m = step.waypoint_mask.unsqueeze(0).to(self.device)

                scores, value = self.model.forward_rl(r_f, w_f, p_f, a_m, r_m, w_m)

                # Get valid logits
                n_r = int(r_m.sum())
                n_w = int(w_m.sum())
                s = scores[0, :n_r, :n_w]
                flat_logits = s.reshape(-1)

                # Skip if all masked (NaN protection)
                if torch.all(torch.isinf(flat_logits) & (flat_logits < 0)):
                    continue

                # Clamp extreme values to prevent overflow
                flat_logits = flat_logits.clamp(min=-50, max=50)

                probs = torch.softmax(flat_logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                new_log_prob = dist.log_prob(
                    torch.tensor(step.action, device=self.device))
                entropy = dist.entropy()

                # PPO clipped surrogate
                adv = all_advantages[idx].to(self.device)
                ratio = torch.exp(new_log_prob - step.log_prob.to(self.device))
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.mcfg.clip_eps,
                                     1 + self.mcfg.clip_eps) * adv
                policy_loss = policy_loss - torch.min(surr1, surr2)

                # Value loss
                ret = all_returns[idx].to(self.device)
                value_loss = value_loss + (value[0] - ret) ** 2

                entropy_sum = entropy_sum + entropy

            mb_size = len(mb_idx)
            loss = (policy_loss / mb_size
                    + self.mcfg.value_loss_coeff * value_loss / mb_size
                    - self.mcfg.entropy_coeff * entropy_sum / mb_size)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += (policy_loss / mb_size).item()
            total_value_loss += (value_loss / mb_size).item()
            total_entropy += (entropy_sum / mb_size).item()
            n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def train(self, scenarios: list,
              checkpoint_dir: str = 'checkpoints/',
              n_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Full RL training loop.

        Args:
            scenarios: list of dicts with keys:
                robot_feats, waypoint_feats, pairwise_feats,
                cost_matrix, wp_to_wp_cost, n_robots, n_waypoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        n_epochs = n_epochs or self.config.n_epochs_rl

        history = {'reward': [], 'policy_loss': [], 'value_loss': []}
        best_reward = -float('inf')

        for epoch in range(n_epochs):
            t0 = time.time()

            # Collect rollouts
            all_steps = []
            all_advantages = []
            all_returns = []
            epoch_rewards = []

            # Sample a batch of scenarios
            batch_size = min(32, len(scenarios))
            batch_indices = np.random.choice(len(scenarios), batch_size, replace=False)

            for s_idx in batch_indices:
                s = scenarios[s_idx]
                steps, ep_reward = self.rollout_episode(
                    s['robot_feats'], s['waypoint_feats'], s['pairwise_feats'],
                    s['cost_matrix'], s['wp_to_wp_cost'],
                    s['n_robots'], s['n_waypoints'],
                )

                advantages, returns = self.compute_gae(
                    steps, self.mcfg.gamma, self.config.gae_lambda
                )

                all_steps.extend(steps)
                all_advantages.append(advantages)
                all_returns.append(returns)
                epoch_rewards.append(ep_reward)

            if not all_steps:
                continue

            all_advantages = torch.cat(all_advantages)
            all_returns = torch.cat(all_returns)

            # Normalize advantages
            if len(all_advantages) > 1:
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            # PPO update (multiple passes over the data)
            for _ in range(4):
                update_metrics = self.ppo_update(all_steps, all_advantages, all_returns)

            elapsed = time.time() - t0
            mean_reward = np.mean(epoch_rewards)
            history['reward'].append(mean_reward)
            history['policy_loss'].append(update_metrics['policy_loss'])
            history['value_loss'].append(update_metrics['value_loss'])

            if (epoch + 1) % self.config.log_interval == 0:
                print(f"RL Epoch {epoch+1}/{n_epochs} "
                      f"reward={mean_reward:.4f} "
                      f"p_loss={update_metrics['policy_loss']:.4f} "
                      f"v_loss={update_metrics['value_loss']:.4f} "
                      f"entropy={update_metrics['entropy']:.4f} "
                      f"time={elapsed:.1f}s")

            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(self.model.state_dict(),
                           os.path.join(checkpoint_dir, 'best_rl.pt'))

            if (epoch + 1) % 50 == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(checkpoint_dir, f'rl_epoch{epoch+1}.pt'))

        torch.save(self.model.state_dict(),
                   os.path.join(checkpoint_dir, 'rl_final.pt'))
        print(f"RL training complete. Best reward: {best_reward:.4f}")

        return history
