
import gymnasium as gym
import gymnasium_robotics 
import numpy as numpy

gym.register_envs(gymnasium_robotics)
env = gym.make("FetchReach-v4", render_mode="human")

obs, info = env.reset()

print(f"Obs type:                {type(obs)}")
print(f"Obs keys:                {obs.keys()}")
print(f"  observation shape:     {obs['observation'].shape}")
print(f"  achieved_goal shape:   {obs['achieved_goal'].shape}")
print(f"  desired_goal shape:    {obs['desired_goal'].shape}")

print(f"Action space:            {env.action_space}")
print(f"Action shape:            {env.action_space.shape}     ← (x, y, z, gripper)")
print(f"Action low:              {env.action_space.low}")
print(f"Action high:             {env.action_space.high}")

print("\nSTEP 5: Running 3 random episodes")
print("-" * 50)

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0
    success = False

    for step in range(50):
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            success = info["is_success"]
            break

    print(f"Episode {episode+1}: "
          f"steps={step+1:3d} | "
          f"total_reward={total_reward:6.1f} | "
          f"success={success}")

print("-" * 50)
print("\nVerification complete. MuJoCo + Gymnasium working.")
env.close()