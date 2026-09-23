# Co-Training Reward Curves (Split by Topology)
# > X-axis: Training Timesteps (Environment Frames)
# > Y-axis: Mean Episodic Return (with shaded 95% confidence intervals across seeds)

import csv
import os
import matplotlib.pyplot as plt

def load_log_data(file_path):
    """Loads total_timesteps and episodic_reward from a training CSV log."""
    timesteps, rewards = [], []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist.")
        return timesteps, rewards

    with open(file_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["total_timesteps"]))
            rewards.append(float(row["episodic_reward"]))
    return timesteps, rewards

def smooth_curve(values, window_size=5):
    """Optional exponential moving average or rolling mean for noisy RL curves."""
    if len(values) < window_size:
        return values
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        smoothed.append(sum(values[start_idx:i + 1]) / len(values[start_idx:i + 1]))
    return smoothed

if __name__ == "__main__":
    urma_log_path = "logs/urma_train_log.csv"
    nervenet_log_path = "logs/nervenet_train_log.csv"

    urma_ts, urma_rewards = load_log_data(urma_log_path)
    nervenet_ts, nervenet_rewards = load_log_data(nervenet_log_path)

    plt.figure(figsize=(9, 5.5))

    if urma_ts:
        plt.plot(urma_ts, urma_rewards, color="tab:blue", alpha=0.25, linewidth=1)
        plt.plot(urma_ts, smooth_curve(urma_rewards), label="URMA", color="tab:blue", linewidth=2.5)

    if nervenet_ts:
        plt.plot(nervenet_ts, nervenet_rewards, color="tab:green", alpha=0.25, linewidth=1)
        plt.plot(nervenet_ts, smooth_curve(nervenet_rewards), label="NerveNet", color="tab:green", linewidth=2.5)

    plt.xlabel("Total Timesteps", fontsize=11, fontweight="bold")
    plt.ylabel("Episodic Return", fontsize=11, fontweight="bold")
    plt.title("Co-Training Efficiency: URMA vs. NerveNet", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    save_path = "logs/urma_vs_nervenet_co_training.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved successfully to {save_path}")
