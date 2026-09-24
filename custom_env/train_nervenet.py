import argparse
from algorithms.nervenet import prepare_inputs, NerveNetActor, NerveNetCritic
from env import BipedEnv
import torch
import os
import csv

def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = BipedEnv()
    obs, _ = env.reset()

    j_obs_dim = obs["j_obs"].shape[-1]
    base_obs_dim = obs["base_obs"].shape[-1]
    target_obs_dim = obs["target_obs"].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = NerveNetActor(
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
        action_dim=1  # 1 scalar output per motor joint
    ).to(device)

    critic = NerveNetCritic(
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


    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    total_timesteps = 0
    log_file_path = "logs/nervenet_train_log.csv"
    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_timesteps", "episodic_reward", "mean_step_reward"])

    print("Initiated actor and critic")
    total_timesteps = 0
    for update_step in range(args.total_updates):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []

        for ep_idx in range(args.episodes_per_update):
            obs, _ = env.reset()
            ep_states, ep_actions, ep_rewards, ep_values, ep_dones, ep_log_probs = [], [], [], [], [], []
            ep_reward = 0.0

            for _ in range(args.rollout_len):
                total_timesteps += 1
                inp = prepare_inputs(obs, env.graph_topology, device)

                with torch.no_grad():
                    act, log_p, _ = actor.get_action_and_log_prob(**inp)
                    val = critic(
                        inp["target_obs"], inp["base_obs"], inp["j_obs"],
                        inp["senders"], inp["receivers"]
                    ).squeeze()

                next_obs, r, term, trunc, _ = env.step(act.reshape(-1).cpu().numpy())
                done = term or trunc

                ep_states.append(inp)
                ep_actions.append(act)
                ep_log_probs.append(log_p.squeeze())
                ep_rewards.append(r)
                ep_values.append(val.squeeze())
                ep_dones.append(done)

                ep_reward += float(r.item() if hasattr(r, "item") else r)
                obs = next_obs
                if done:
                    break

            current_ep_num = update_step * args.episodes_per_update + ep_idx

            with open(log_file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_ep_num, total_timesteps, ep_reward, ep_reward / len(ep_rewards)])

            if update_step % 5 == 0 and current_ep_num % 10 == 0:
                print(f"Update: {update_step} | Ep: {current_ep_num} | Timesteps: {total_timesteps} | Reward: {ep_reward:.2f}")

            returns, R = [], 0
            for r, d in zip(reversed(ep_rewards), reversed(ep_dones)):
                R = r + args.gamma * R * (1 - float(d))
                returns.insert(0, R)

            ep_returns = torch.tensor(returns, dtype=torch.float32).to(device)
            ep_values_tensor = torch.stack(ep_values)
            ep_advantages = ep_returns - ep_values_tensor

            # append to global update buffer
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_log_probs.extend(ep_log_probs)
            all_returns.append(ep_returns)
            all_advantages.append(ep_advantages)

        flat_returns = torch.cat(all_returns, dim=0).to(device)
        flat_advantages = torch.cat(all_advantages, dim=0).to(device)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        obs_keys = {"target_obs", "base_obs", "j_obs"}
        batch_inp = {
            k: (
                torch.cat([s[k] for s in all_states], dim=0).to(device)
                if k in obs_keys
                else all_states[0][k]
            )
            for k in all_states[0].keys()
        }
        batch_actions = torch.cat(all_actions, dim=0).to(device)
        old_log_probs = torch.stack(all_log_probs).to(device)

        for _ in range(args.epochs):
            _, new_lp, entropy = actor.get_action_and_log_prob(**batch_inp, action=batch_actions)
            v = critic(
                batch_inp["target_obs"], batch_inp["base_obs"], batch_inp["j_obs"],
                batch_inp["senders"], batch_inp["receivers"]
            ).squeeze()

            ratio = torch.exp(new_lp - old_log_probs)
            surr1 = ratio * flat_advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (v - flat_returns).pow(2).mean()
            entropy_loss = -0.01 * entropy.mean()

            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    env.close()

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NerveNet policy")
    parser.add_argument("--save-path", type=str, default="checkpoints/nervenet_checkpoint.pth", help="Path to model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--total-updates", type=int, default=10, help="Number of updates")
    parser.add_argument("--episodes-per-update", type=int, default=5, help="Number of episodes per update")
    parser.add_argument("--rollout-len", type=int, default=100, help="Length of single rollout")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()
    train(args)
