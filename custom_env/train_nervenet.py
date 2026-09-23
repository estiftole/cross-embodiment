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
    )

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
    )


    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    total_timesteps = 0
    log_file_path = "logs/nervenet_train_log.csv"
    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_timesteps", "episodic_reward", "mean_step_reward"])

    history = {"episode": [], "timesteps": [], "reward": []}
    print("Initiated actor and critic")
    for episode in range(args.episodes):
        states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []
        ep_reward = 0.0

        for _ in range(args.rollout_len):
            total_timesteps += 1

            inp = prepare_inputs(obs, env.graph_topology)
            with torch.no_grad():
                act, log_p, _ = actor.get_action_and_log_prob(**inp)
                val = critic(
                    inp["target_obs"],
                    inp["base_obs"],
                    inp["j_obs"],
                    inp["senders"],
                    inp["receivers"]
                ).squeeze()

            next_obs, r, term, trunc, _ = env.step(act.reshape(-1).cpu().numpy())
            done = term or trunc

            states.append(inp)
            actions.append(act)
            log_probs.append(log_p)
            rewards.append(r)
            values.append(val.squeeze())
            dones.append(done)

            ep_reward += float(r.item() if hasattr(r, "item") else r)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        # Log metrics for this episode
        history["episode"].append(episode)
        history["timesteps"].append(total_timesteps)
        history["reward"].append(ep_reward)

        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_timesteps, ep_reward, ep_reward / args.rollout_len])

        print(f"Episode: {episode} | Timesteps: {total_timesteps} | Reward: {ep_reward:.2f}")

        returns, R = [], 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + args.gamma * R * (1 - float(d))
            returns.insert(0, R)

        returns = torch.tensor(returns)
        advantages = returns - torch.tensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.epochs):
            for i in range(args.rollout_len):
                inp = states[i]
                act = actions[i]
                old_lp = log_probs[i]
                adv = advantages[i]
                ret = returns[i]

                _, new_lp, entropy = actor.get_action_and_log_prob(**inp, action=act)
                v = critic(
                    inp["target_obs"],
                    inp["base_obs"],
                    inp["j_obs"],
                    inp["senders"],
                    inp["receivers"]
                ).squeeze()

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

                policy_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * (v - ret).pow(2)
                entropy_loss = -0.01 * entropy

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

    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train for")
    parser.add_argument("--rollout-len", type=int, default=100, help="Length of single rollout")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()
    train(args)
