import argparse
import os
import csv
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import EnvBase, Transform, TransformedEnv, SerialEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, ListStorage
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from algorithms.nervenet import prepare_inputs, NerveNetActor, NerveNetCritic
from env import CrossEmbodimentEnv


class NerveNetActorWrapper(nn.Module):
    """Wraps NerveNetActor to accept dict inputs and return distribution parameters."""
    def __init__(self, actor_net):
        super().__init__()
        self.actor_net = actor_net

    def forward(self, target_obs, base_obs, j_obs, senders, receivers):
        # Forward pass returning mean and scale for TanhNormal distribution
        loc, scale, _ = self.actor_net.get_action_and_log_prob(
            target_obs=target_obs,
            base_obs=base_obs,
            j_obs=j_obs,
            senders=senders,
            receivers=receivers,
        )
        return loc, scale


class NerveNetCriticWrapper(nn.Module):
    """Wraps NerveNetCritic to output standard state-values."""
    def __init__(self, critic_net):
        super().__init__()
        self.critic_net = critic_net

    def forward(self, target_obs, base_obs, j_obs, senders, receivers):
        value = self.critic_net(
            target_obs=target_obs,
            base_obs=base_obs,
            j_obs=j_obs,
            senders=senders,
            receivers=receivers,
        )
        return value.squeeze(-1)


def make_env(starting_embodiment="biped"):
    raw_env = CrossEmbodimentEnv(starting_embodiment=starting_embodiment)
    return raw_env


def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_env = make_env(starting_embodiment="biped")
    obs, _ = dummy_env.reset()

    j_obs_dim = obs["j_obs"].shape[-1]
    base_obs_dim = obs["base_obs"].shape[-1]
    target_obs_dim = obs["target_obs"].shape[-1]
    dummy_env.close()

    raw_actor = NerveNetActor(
        j_obs_dim=j_obs_dim,
        target_obs_dim=target_obs_dim,
        base_obs_dim=base_obs_dim,
        obs_enc_hidden_dim=32,
        hidden_state_dim=64,
        updater_hidden_dim=64,
        msg_hidden_dim=32,
        msg_dim=16,
        iterations=2,
        dec_hidden_dim=32,
        action_dim=1,
    ).to(device)

    raw_critic = NerveNetCritic(
        j_obs_dim=j_obs_dim,
        target_obs_dim=target_obs_dim,
        base_obs_dim=base_obs_dim,
        obs_enc_hidden_dim=32,
        hidden_state_dim=64,
        updater_hidden_dim=64,
        msg_hidden_dim=32,
        msg_dim=16,
        iterations=2,
        dec_hidden_dim=32,
    ).to(device)

    in_keys = ["target_obs", "base_obs", "j_obs", "senders", "receivers"]

    actor_wrapper = NerveNetActorWrapper(raw_actor)
    actor_td_module = TensorDictModule(
        module=actor_wrapper,
        in_keys=in_keys,
        out_keys=["loc", "scale"],
    )
    policy_module = ProbabilisticActor(
        module=actor_td_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    ).to(device)

    critic_wrapper = NerveNetCriticWrapper(raw_critic)
    value_module = ValueOperator(
        module=critic_wrapper,
        in_keys=in_keys,
        out_keys=["state_value"],
    ).to(device)

    advantage_module = GAE(
        gamma=args.gamma,
        l1=0.95,
        value_network=value_module,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        loss_critic_type="smooth_l1",
    )

    optimizer = torch.optim.Adam(
        loss_module.parameters(), lr=args.lr, eps=1e-5
    )

    env_fn = lambda: make_env(starting_embodiment="biped")
    collector = SyncDataCollector(
        create_env_fn=env_fn,
        policy=policy_module,
        frames_per_batch=args.rollout_len * args.episodes_per_update,
        total_frames=args.total_updates * args.rollout_len * args.episodes_per_update,
        device=device,
    )

    replay_buffer = ReplayBuffer(storage=ListStorage(max_size=1000))

    log_file_path = "logs/nervenet_train_log.csv"
    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "total_frames", "mean_reward", "ppo_loss"])

    print("Initiated TorchRL NerveNet Training Pipeline.")

    total_frames = 0
    for update_idx, tensordict_data in enumerate(collector):
        total_frames += tensordict_data.numel()

        with torch.no_grad():
            advantage_module(tensordict_data)

        replay_buffer.extend(tensordict_data.reshape(-1))

        # PPO Optimization Epochs
        for epoch in range(args.epochs):
            for _ in range(tensordict_data.numel() // 64):  # Batch size = 64
                subdata = replay_buffer.sample(64).to(device)

                loss_vals = loss_module(subdata)
                total_loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()

        replay_buffer.empty()

        # Log Metrics
        mean_reward = tensordict_data["next", "reward"].mean().item()
        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([update_idx, total_frames, mean_reward, total_loss.item()])

        if update_idx % 10 == 0:
            print(
                f"Update: {update_idx} | Total Frames: {total_frames} | "
                f"Mean Step Reward: {mean_reward:.3f} | Loss: {total_loss.item():.4f}"
            )

    collector.shutdown()

    # 8. Checkpoint Export
    checkpoint = {
        "actor_state_dict": raw_actor.state_dict(),
        "critic_state_dict": raw_critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "j_obs_dim": j_obs_dim,
            "target_obs_dim": target_obs_dim,
            "base_obs_dim": base_obs_dim,

            "obs_enc_hidden_dim": 32,
            "hidden_state_dim": 64,
            "updater_hidden_dim": 64,
            "msg_hidden_dim": 32,
            "msg_dim": 16,
            "iterations": 2,
            "dec_hidden_dim": 32,
            "action_dim": 1
        }
    }
    torch.save(checkpoint, args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NerveNet policy with TorchRL")
    parser.add_argument("--save-path", type=str, default="checkpoints/nervenet_checkpoint.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=5)
    parser.add_argument("--rollout-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(args)
