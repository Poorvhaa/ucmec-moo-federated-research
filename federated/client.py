"""
Federated Multi-Agent Learning: Independent Edge Site Client.
"""
import copy
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from envs.env_discrete import DiscreteActionEnv
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from algorithms.algorithm.r_mappo import RMAPPO
from utils.shared_buffer import SharedReplayBuffer
from federated.scenarios import ClientScenarioConfig
from federated.core import check_finite_tensors
from federated.accounting import calculate_tensor_payload_bytes


def _t2n(x):
    return x.detach().cpu().numpy()


class FederatedClient:
    """
    Simulates an independent edge site containing its own:
    - Local multi-agent MEC environment and independent PRNG
    - Shared/role-specific actor network
    - Personalized local centralized critic and local value normalizer
    - Local replay buffer for fresh on-policy rollout collection
    - Local optimizer states
    """

    def __init__(
        self,
        config: ClientScenarioConfig,
        all_args: Any,
        device: torch.device = torch.device("cpu"),
        preset_name: str = "balanced",
        moo_weights: Optional[Dict[str, float]] = None,
        use_corrected_cpu_alloc: bool = True,
        solver_name: str = "CLARABEL",
        channel_mode: str = "legacy",
    ):
        self.config = config
        self.client_id = config.client_id
        self.site_name = config.site_name
        self.seed = config.seed
        self.device = device
        self.all_args = copy.deepcopy(all_args)

        # Initialize environment with client-specific physical & random configuration
        self.env = DiscreteActionEnv(
            preset_name=preset_name,
            moo_weights=moo_weights,
            use_corrected_cpu_alloc=use_corrected_cpu_alloc,
            solver_name=solver_name,
            physical_params={"channel_mode": channel_mode},
            max_steps=config.local_rollout_steps,
            task_size_range=config.task_size_range,
            c_edge_range=config.c_edge_range,
            epsilon=config.epsilon,
            seed=config.seed,
            **config.extra_env_kwargs,
        )
        self.env.seed(self.seed)

        self.num_agents = self.env.num_agent
        self.obs_space = self.env.observation_space[0]
        self.share_obs_space = (
            self.env.share_observation_space[0]
            if getattr(all_args, "use_centralized_V", True)
            else self.obs_space
        )
        self.act_space = self.env.action_space[0]

        # Policy: Actor + Personalized Local Critic
        self.policy = RMAPPOPolicy(
            self.all_args,
            self.obs_space,
            self.share_obs_space,
            self.act_space,
            device=self.device,
        )

        # Local PPO Trainer
        self.trainer = RMAPPO(self.all_args, self.policy, device=self.device)

        # Local Replay Buffer (single-process rollout: n_rollout_threads=1)
        self.buffer_args = copy.deepcopy(all_args)
        self.buffer_args.n_rollout_threads = 1
        self.buffer_args.episode_length = config.local_rollout_steps

        self.buffer = SharedReplayBuffer(
            self.buffer_args,
            self.num_agents,
            self.obs_space,
            self.share_obs_space,
            self.act_space,
        )

        self.current_obs: Optional[np.ndarray] = None
        self.last_metrics: Dict[str, float] = {}

    def receive_global_actor(
        self,
        global_actor_state: Dict[str, torch.Tensor],
        reset_optimizer: bool = True,
    ):
        """
        Receives versioned global actor weights from the coordinator.
        - Loads weights into local actor network.
        - Resets actor Adam optimizer state if reset_optimizer=True.
        - Discards any obsolete on-policy trajectory buffer data.
        - Strictly preserves local critic parameters, critic optimizer, and value normalizer.
        """
        # Load actor weights
        self.policy.actor.load_state_dict(
            {k: v.to(device=self.device) for k, v in global_actor_state.items()}
        )

        # Clear obsolete buffer data
        self.buffer.step = 0

        # Reset actor optimizer momentum if configured
        if reset_optimizer:
            self.policy.actor_optimizer.state.clear()

    def collect_fresh_rollouts(
        self,
        rollout_steps: Optional[int] = None,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Interacts with local multi-agent environment using the current actor policy
        to collect fresh on-policy transitions. Returns valid sample count and rollout metrics.
        """
        steps_to_collect = rollout_steps or self.config.local_rollout_steps
        self.buffer.episode_length = steps_to_collect
        self.buffer.step = 0

        # Reset environment
        obs = self.env.reset()
        share_obs = obs.reshape(-1)
        self.current_obs = obs

        # Initialize buffer step 0
        self.buffer.share_obs[0] = np.expand_dims(np.repeat(share_obs[None, :], self.num_agents, axis=0), 0)
        self.buffer.obs[0] = np.expand_dims(obs, 0)
        self.buffer.masks[0] = np.ones((1, self.num_agents, 1), dtype=np.float32)
        self.buffer.rnn_states[0] = np.zeros((1, self.num_agents, 1, self.all_args.hidden_size), dtype=np.float32)
        self.buffer.rnn_states_critic[0] = np.zeros_like(self.buffer.rnn_states[0])

        step_delays = []
        step_energies = []
        step_overheads = []
        step_successes = []
        step_rewards = []

        valid_transitions = 0

        self.trainer.prep_rollout()

        for step in range(steps_to_collect):
            with torch.no_grad():
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                ) = self.policy.get_actions(
                    np.concatenate(self.buffer.share_obs[step]),
                    np.concatenate(self.buffer.obs[step]),
                    np.concatenate(self.buffer.rnn_states[step]),
                    np.concatenate(self.buffer.rnn_states_critic[step]),
                    np.concatenate(self.buffer.masks[step]),
                )

            values = np.array(np.split(_t2n(values), 1))
            actions = np.array(np.split(_t2n(actions), 1))
            action_log_probs = np.array(np.split(_t2n(action_log_probs), 1))
            rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), 1))
            rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 1))

            # Convert action indices to one-hot for environment
            actions_env = []
            for a_idx in range(self.num_agents):
                act_val = int(actions[0, a_idx, 0])
                one_hot = np.zeros(self.act_space.n, dtype=np.float32)
                one_hot[act_val] = 1.0
                actions_env.append(one_hot)
            actions_env = np.array(actions_env)

            # Step environment
            next_obs, rews, dones, infos = self.env.step(actions_env)

            for a_info in infos:
                if isinstance(a_info, dict):
                    step_delays.append(a_info.get("delay_ms", 0.0))
                    step_energies.append(a_info.get("energy_mj", 0.0))
                    step_overheads.append(a_info.get("overhead_bytes", 0.0))
                    step_successes.append(1.0 if a_info.get("task_success", False) else 0.0)
                    step_rewards.append(a_info.get("individual_reward", 0.0))

            valid_transitions += self.num_agents

            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
            if np.all(dones):
                masks = np.zeros((1, self.num_agents, 1), dtype=np.float32)

            share_next_obs = next_obs.reshape(-1)
            share_next_obs_rep = np.expand_dims(
                np.repeat(share_next_obs[None, :], self.num_agents, axis=0), 0
            )

            # Insert step into buffer
            self.buffer.insert(
                share_obs=share_next_obs_rep,
                obs=np.expand_dims(next_obs, 0),
                rnn_states_actor=rnn_states_actor,
                rnn_states_critic=rnn_states_critic,
                actions=actions,
                action_log_probs=action_log_probs,
                value_preds=values,
                rewards=np.expand_dims(rews.reshape(self.num_agents, 1), 0),
                masks=masks,
            )

            if np.all(dones):
                obs = self.env.reset()
                self.buffer.masks[self.buffer.step] = np.ones((1, self.num_agents, 1), dtype=np.float32)

        # Compute returns with local personalized critic
        self.trainer.prep_rollout()
        with torch.no_grad():
            next_values = self.policy.get_values(
                np.concatenate(self.buffer.share_obs[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
        next_values = np.array(np.split(_t2n(next_values), 1))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

        mean_overhead_b = float(np.mean(step_overheads)) if step_overheads else 0.0
        metrics = {
            "reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
            "delay_ms": float(np.mean(step_delays)) if step_delays else 0.0,
            "energy_mj": float(np.mean(step_energies)) if step_energies else 0.0,
            "overhead_kb": float(mean_overhead_b / 1024.0),
            "task_success_rate": float(np.mean(step_successes)) if step_successes else 0.0,
        }
        self.last_metrics = metrics
        return valid_transitions, metrics

    def perform_local_updates(self, local_epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Executes configured local PPO updates on the fresh local rollout buffer.
        """
        self.trainer.prep_training()
        epochs = local_epochs or self.config.local_epochs
        old_epochs = self.trainer.ppo_epoch
        self.trainer.ppo_epoch = epochs

        train_infos = self.trainer.train(self.buffer)
        self.trainer.ppo_epoch = old_epochs

        # Clear buffer after update
        self.buffer.after_update()

        metrics = {
            "policy_loss": float(train_infos.get("policy_loss", 0.0)),
            "value_loss": float(train_infos.get("value_loss", 0.0)),
            "dist_entropy": float(train_infos.get("dist_entropy", 0.0)),
            "actor_grad_norm": float(train_infos.get("actor_grad_norm", 0.0)),
            "critic_grad_norm": float(train_infos.get("critic_grad_norm", 0.0)),
        }
        return metrics

    def package_submission(
        self,
        round_idx: int,
        base_version: int,
        valid_samples: int,
        runtime_s: float,
        train_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Packages actor parameters and minimal scalar metadata.
        Raw trajectories (observations, actions, rewards) are strictly excluded.
        """
        # Extract actor parameters as CPU tensors
        actor_state = {
            k: v.detach().cpu().clone()
            for k, v in self.policy.actor.state_dict().items()
        }

        if not check_finite_tensors(actor_state):
            raise ValueError(f"Client {self.client_id} generated non-finite parameters.")

        payload_bytes = calculate_tensor_payload_bytes(actor_state)

        combined_metrics = {**self.last_metrics, **train_metrics}

        return {
            "client_id": self.client_id,
            "site_name": self.site_name,
            "round": round_idx,
            "base_version": base_version,
            "valid_samples": valid_samples,
            "runtime_s": runtime_s,
            "actor_state": actor_state,
            "metrics": combined_metrics,
            "payload_bytes": payload_bytes,
        }
