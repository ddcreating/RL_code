import gymnasium as gym
from stable_baselines3 import SAC
import imageio

model = SAC.load("./sac_local/sac_models/sac_hopper/best_model.zip")

# Try "Follow Camera"
env = gym.make("Hopper-v4", render_mode="rgb_array", camera_name="track")  # If an error occurs, use camera_id
# env = gym.make("Hopper-v4", render_mode="rgb_array", camera_id=0)

frames = []
obs, _ = env.reset()
for _ in range(600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

imageio.mimsave("hopper_track.gif", frames, fps=30)
print("Saved hopper_track.gif")
