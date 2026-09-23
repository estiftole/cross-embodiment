import argparse
import torch
import numpy as np
from pathlib import Path
from algorithms.urma import URMAActor
from env import BipedEnv

from gymnasium.wrappers import RecordVideo, TimeLimit

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate(args):
    set_seed(args.seed)
    render_mode = "human" if args.render else "rgb_array"

    env = BipedEnv(render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    if render_mode == "rgb_array":
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder=out_dir,
            # name_prefix="urma",
            episode_trigger=lambda episode_id: True,
        )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    actor = URMAActor(**checkpoint["config"])
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.to(args.device)
    actor.eval()

    print(f"Loaded actor from {args.checkpoint} onto {args.device}")


    for episode in range(args.episodes):
        print(f"Episode {episode}")
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        step = 0

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
                act, log_prob = actor(**inp)

            action_np = act.squeeze(0).cpu().numpy().reshape(-1)

            next_obs, reward, term, trunc, _ = env.step(action_np)
            done = term or trunc

            step += 1
            obs = next_obs

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate URMA Actor and collect data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/urma_checkpoint.pth", help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda)")
    parser.add_argument("--render", action="store_true", help="Enable environment rendering")
    parser.add_argument("--output-dir", type=str, default="./videos/urma", help="Directory to save videos")

    args = parser.parse_args()
    evaluate(args)
