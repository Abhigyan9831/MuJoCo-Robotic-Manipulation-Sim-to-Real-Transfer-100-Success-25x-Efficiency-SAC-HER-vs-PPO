import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


gym.register_envs(gymnasium_robotics)

env = make_vec_env("FetchReach-v4", n_envs=4)
eval_env = make_vec_env("FetchReach-v4", n_envs=1)

model = PPO("MultiInputPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=1, tensorboard_log=LOGS_DIR, device="cuda") 



eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

print("Starting training")
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)

model.save(os.path.join(MODELS_DIR, "ppo_fetchreach_final"))
print("Training complete. Model saved to", MODELS_DIR)