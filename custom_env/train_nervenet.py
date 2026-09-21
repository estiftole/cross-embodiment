from algorithms.nervenet.nervenet import prepare_inputs, NerveNetActor, NerveNetCritic
from env import BipedEnv
import numpy as np

import torch
from torch.distributions import Normal


if __name__ == "__main__":
    env = BipedEnv(render_mode="human")
    obs, info = env.reset()

    epochs = 10
    gamma = 0.99
    lr = 3e-4
    num_episodes = 10
    rollout_len = 100

    base_obs_dim = obs["base_obs"].shape[-1]
    target_obs_dim = obs["target_obs"].shape[-1]
    j_obs_dim = obs["joint_obs"].shape[-1]

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
        action_dec_hidden_dim=32,
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
        value_dec_hidden_dim=32
    )

    log_std = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=lr)

    print("Initiated actor and critic")
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        for _ in range(rollout_len):
            inp = prepare_inputs(obs, env.graph_topology)
            with torch.no_grad():
                mu = actor(**inp).squeeze()
                val = critic(inp["target_obs"], inp["base_obs"], inp["j_obs"], inp["senders"], inp["receivers"])
                dist = Normal(mu, torch.exp(log_std))
                act = dist.sample()
                log_p = dist.log_prob(act).sum()

            next_obs, r, term, trunc, info = env.step(act.squeeze(0).cpu().numpy())
            done = term or trunc

            states.append(inp)
            actions.append(act)
            log_probs.append(log_p)
            rewards.append(r)
            values.append(val.squeeze())
            dones.append(done)

            obs = next_obs
            if done:
                obs, info = env.reset()

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
                inp, act, old_lp, adv, ret = states[i], actions[i], log_probs[i], advantages[i], returns[i]

                mu = actor(**inp)
                v = critic(inp["target_obs"], inp["base_obs"], inp["j_obs"], inp["senders"], inp["receivers"]).squeeze()

                dist = Normal(mu, torch.exp(log_std))
                new_lp = dist.log_prob(act).sum()

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

                loss = -torch.min(surr1, surr2) + 0.5 * (v - ret).pow(2) - 0.01 * dist.entropy().sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    env.close()
