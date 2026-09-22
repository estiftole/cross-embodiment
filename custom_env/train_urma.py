from algorithms.urma import URMAActor, URMACritic
from env import BipedEnv
import torch


if __name__ == "__main__":
    env = BipedEnv()
    obs, info = env.reset()

    epochs = 10
    num_episodes = 1
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

    print("Initiated actor and critic")
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        states, actions, rewards, values, dones = [], [], [], [], []

        for _ in range(rollout_len):
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
                act = actor(**inp).squeeze()
                val = critic(**inp).squeeze()

            next_obs, r, term, trunc, info = env.step(act.cpu().numpy())
            done = term or trunc

            states.append(inp)
            actions.append(act)
            rewards.append(r)
            values.append(val)
            dones.append(done)

            obs = next_obs
            if done:
                obs, info = env.reset()

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

                v = critic(**inp).squeeze()
                value_loss = 0.5 * (v - ret).pow(2)

                optimizer.zero_grad()
                value_loss.backward()
                optimizer.step()

    env.close()

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
