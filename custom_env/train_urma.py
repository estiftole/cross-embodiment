from algorithms.urma import URMAActor, URMACritic
from env import BipedEnv
import torch
import csv
import os

import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # env = BipedEnv(render_mode="human")
    env = BipedEnv()
    obs, info = env.reset()

    epochs = 5
    num_episodes = 10
    rollout_len = 100
    gamma = 0.99
    lr = 3e-4

    j_obs_dim = obs["j_obs"].shape[-1]
    j_desc_dim = obs["j_desc"].shape[-1]
    ee_obs_dim = obs["ee_obs"].shape[-1]
    ee_desc_dim = obs["ee_desc"].shape[-1]
    base_obs_dim = obs["base_obs"].shape[-1]
    target_obs_dim = obs["target_obs"].shape[-1]

    actor = URMAActor(
        j_obs_dim=j_obs_dim,
        j_desc_dim=j_desc_dim,
        ee_obs_dim=ee_obs_dim,
        ee_desc_dim=ee_desc_dim,
        base_obs_dim=base_obs_dim,
        target_obs_dim=target_obs_dim,
        enc_hidden_dim=32,
        j_latent_dim=64,
        ee_latent_dim=64,
        embed_dim=64,
        attn_heads=4,
        hidden_dim=128,
        action_latent_dim=64,
        dec_hidden_dim=64,
        dec_out_dim=64,
        mu_hidden_dim=64,
        action_dim=1
    )

    critic = URMACritic(
        j_obs_dim=j_obs_dim,
        j_desc_dim=j_desc_dim,
        ee_obs_dim=ee_obs_dim,
        ee_desc_dim=ee_desc_dim,
        base_obs_dim=base_obs_dim,
        target_obs_dim=target_obs_dim,
        enc_hidden_dim=32,
        j_latent_dim=64,
        ee_latent_dim=64,
        embed_dim=64,
        attn_heads=4,
        hidden_dim=128
    )

    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    total_timesteps = 0
    log_file_path = "logs/urma_train_log.csv"
    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_timesteps", "episodic_reward", "mean_step_reward"])

    history = {"episode": [], "timesteps": [], "reward": []}
    print("Initiated actor and critic")
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []
        ep_reward = 0.0

        for _ in range(rollout_len):
            total_timesteps += 1
            inp = {
                "target_obs": torch.as_tensor(obs["target_obs"], dtype=torch.float32).unsqueeze(0),
                "base_obs": torch.as_tensor(obs["base_obs"], dtype=torch.float32).unsqueeze(0),
                "j_obs": torch.as_tensor(obs["j_obs"], dtype=torch.float32).unsqueeze(0),
                "j_desc": torch.as_tensor(obs["j_desc"], dtype=torch.float32).unsqueeze(0),
                "ee_obs": torch.as_tensor(obs["ee_obs"], dtype=torch.float32).unsqueeze(0),
                "ee_desc": torch.as_tensor(obs["ee_desc"], dtype=torch.float32).unsqueeze(0),
            }
            # inp = obs

            with torch.no_grad():
                act, log_prob = actor(**inp)
                act = act.squeeze()
                val = critic(**inp).squeeze()

            next_obs, r, term, trunc, info = env.step(act.cpu().numpy())
            done = term or trunc

            log_probs.append(log_prob)
            states.append(inp)
            actions.append(act)
            rewards.append(r)
            values.append(val)
            dones.append(done)

            ep_reward += float(r.item() if hasattr(r, "item") else r)
            obs = next_obs
            if done:
                obs, info = env.reset()

        # Log metrics for this episode
        history["episode"].append(episode)
        history["timesteps"].append(total_timesteps)
        history["reward"].append(ep_reward)

        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_timesteps, ep_reward, ep_reward / rollout_len])

        print(f"Episode: {episode} | Timesteps: {total_timesteps} | Reward: {ep_reward:.2f}")

        returns, R = [], 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - float(d))
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for i in range(rollout_len):
                inp = states[i]
                ret = returns[i]
                old_action = actions[i]
                adv = advantages[i]

                v = critic(**inp).squeeze()
                value_loss = 0.5 * (v - ret).pow(2)

                _, new_log_prob = actor(**inp, action=old_action)
                actor_loss = -(new_log_prob * adv)

                total_loss = (value_loss + actor_loss).mean()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    env.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["timesteps"], history["reward"], label="URMA (Biped)", color="tab:blue", linewidth=2)
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episodic Return")
    plt.title("URMA Co-Training Reward Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = "logs/urma_reward_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {plot_path}")

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "config": {
            "j_obs_dim": j_obs_dim,
            "j_desc_dim": j_desc_dim,
            "ee_obs_dim": ee_obs_dim,
            "ee_desc_dim": ee_desc_dim,
            "base_obs_dim": base_obs_dim,
            "target_obs_dim": target_obs_dim,
            "action_dim": 1,

            "enc_hidden_dim": 32,
            "j_latent_dim": 64,
            "ee_latent_dim": 64,
            "embed_dim": 64,
            "attn_heads": 4,
            "hidden_dim": 128,
            "action_latent_dim": 64,
            "dec_hidden_dim": 64,
            "dec_out_dim": 64,
            "mu_hidden_dim": 64,
        }
    }
    save_path = "checkpoints/urma_checkpoint.pth"
    torch.save(checkpoint, save_path)
