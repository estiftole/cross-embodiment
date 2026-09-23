from algorithms.nervenet import prepare_inputs, NerveNetActor, NerveNetCritic
from env import BipedEnv
import torch
import os
import csv
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


    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    total_timesteps = 0
    log_file_path = "logs/nervenet_train_log.csv"
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

            next_obs, r, term, trunc, info = env.step(act.reshape(-1).cpu().numpy())
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

        returns = torch.tensor(returns)
        advantages = returns - torch.tensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for i in range(rollout_len):
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

    plt.figure(figsize=(8, 5))
    plt.plot(history["timesteps"], history["reward"], label="NerveNet (Biped)", color="tab:blue", linewidth=2)
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episodic Return")
    plt.title("NerveNet Co-Training Reward Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = "logs/nervenet_reward_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {plot_path}")


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
    save_path = "checkpoints/nervenet_checkpoint.pth"
    torch.save(checkpoint, save_path)
