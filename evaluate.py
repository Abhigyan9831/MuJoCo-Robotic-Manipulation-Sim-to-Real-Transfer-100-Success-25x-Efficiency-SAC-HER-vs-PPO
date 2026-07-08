# evaluate.py
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, SAC
import numpy as np
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

gym.register_envs(gymnasium_robotics)
eval_env = gym.make("FetchReach-v4", render_mode="human")

print("Loading model")
model = SAC.load(os.path.join(MODELS_DIR, "best_model_temp_now"), env=eval_env)



n_episodes      = 100
successes       = []
distances       = []
inference_times = []

print(f"Running {n_episodes} evaluation episodes")
print("-" * 50)

for episode in range(n_episodes):
    obs, info = eval_env.reset()
    episode_success = False
    episode_distances = []

    for step in range(50):
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        end = time.perf_counter()
        inference_times.append((end - start) * 1000)

        obs, reward, terminated, truncated, info = eval_env.step(action)

        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        episode_distances.append(dist)

        if terminated or truncated:
            episode_success = info["is_success"]
            break

    successes.append(float(episode_success))
    distances.append(min(episode_distances))

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1:3d}/100 | "
              f"success={episode_success} | "
              f"best_dist={min(episode_distances):.4f}m")

print("-" * 50)
print(f"\n{'='*40}")
print(f"EVALUATION RESULTS (100 episodes)")
print(f"{'='*40}")
print(f"Success Rate:        {np.mean(successes)*100:.1f}%")
print(f"Avg Best Distance:   {np.mean(distances)*1000:.1f}mm")
print(f"Avg Inference Time:  {np.mean(inference_times):.2f}ms per step")
print(f"Max Inference Time:  {np.max(inference_times):.2f}ms per step")
print(f"Inference FPS:       {1000/np.mean(inference_times):.0f} FPS")
print(f"{'='*40}\n")

eval_env.close()