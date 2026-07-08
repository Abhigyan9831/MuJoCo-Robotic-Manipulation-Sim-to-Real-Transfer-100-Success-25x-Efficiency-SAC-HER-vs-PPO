
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import EvalCallback
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

gym.register_envs(gymnasium_robotics)


env = gym.make("FetchReach-v4")
eval_env = gym.make("FetchReach-v4")

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,        
        goal_selection_strategy="future",  
    ),
    learning_rate=1e-3,
    buffer_size=1_000_000,
    learning_starts=1000,        
    batch_size=256,
    gamma=0.95,
    verbose=1,
    tensorboard_log=LOGS_DIR,
    device="cuda"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR,
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

print("Starting SAC+HER ")
model.learn(
    total_timesteps=200_000,     
    callback=eval_callback
)

model.save(os.path.join(MODELS_DIR, "sac_her_fetchreach_final"))
print("Training complete. Model saved to", MODELS_DIR)