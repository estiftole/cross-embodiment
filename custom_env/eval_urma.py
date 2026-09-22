import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from algorithms.urma import URMAActor
from env import BipedEnv

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate(args):
    set_seed(args.seed)

    render_mode = "human" if args.render else "rgb_array"
    env = BipedEnv(render_mode=render_mode)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    actor = URMAActor(**checkpoint["config"])
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.to(args.device)
    actor.eval()

    print(f"Loaded actor from {args.checkpoint} onto {args.device}")

    episode_stats = []
    trajectory_data = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < args.max_steps:
            inp = {
                "target_obs": torch.as_tensor(obs["target_obs"], dtype=torch.float32).unsqueeze(0),
                "base_obs": torch.as_tensor(obs["base_obs"], dtype=torch.float32).unsqueeze(0),
                "j_obs": torch.as_tensor(obs["j_obs"], dtype=torch.float32).unsqueeze(0),
                "j_desc": torch.as_tensor(obs["j_desc"], dtype=torch.float32).unsqueeze(0),
                "ee_obs": torch.as_tensor(obs["ee_obs"], dtype=torch.float32).unsqueeze(0),
                "ee_desc": torch.as_tensor(obs["ee_desc"], dtype=torch.float32).unsqueeze(0),
            }
            inp = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}

            with torch.no_grad():
                act = actor(**inp).squeeze()

            action_np = act.reshape(-1).cpu().numpy()

            next_obs, reward, term, trunc, _ = env.step(action_np)
            done = term or trunc

            if args.save_trajectories:
                trajectory_data.append({
                    "episode": episode,
                    "step": step,
                    "reward": reward,
                    "action_magnitude": np.linalg.norm(action_np),
                    # "obs_x": obs[0], # + specific state dimensions if analyzing specific physical variables
                })

            ep_reward += reward
            step += 1
            obs = next_obs

        episode_stats.append({
            "episode": episode,
            "return": ep_reward,
            "length": step,
            "terminated": term,
            "truncated": trunc
        })

        print(f"Episode {episode+1:03d} | Return: {ep_reward:8.2f} | Steps: {step}")

    env.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_episodes = pd.DataFrame(episode_stats)
    df_episodes.to_csv(out_dir / "episode_summaries.csv", index=False)

    if args.save_trajectories:
        df_traj = pd.DataFrame(trajectory_data)
        df_traj.to_csv(out_dir / "trajectory_details.csv", index=False)

    print(f"\nEvaluation complete. Data saved to {out_dir}/")
    print(f"Average Return: {df_episodes['return'].mean():.2f} ± {df_episodes['return'].std():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate URMA Actor and collect data")
    parser.add_argument("--checkpoint", type=str, default="urma_checkpoint.pth", help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda)")
    parser.add_argument("--render", action="store_true", help="Enable environment rendering")
    parser.add_argument("--output-dir", type=str, default="./eval_data", help="Directory to save CSV outputs")
    parser.add_argument("--save-trajectories", action="store_true", help="Save step-by-step state/action data for behavioral testing")

    args = parser.parse_args()
    evaluate(args)
