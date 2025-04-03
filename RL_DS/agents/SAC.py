from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from RL_DS.envs.retailer_gym import RetailerOrdersEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from RL_DS.utils.norm2 import MinMaxWrapper

# Wrap your environment if needed (e.g., normalization)
env = RetailerOrdersEnv(time_horizon=47, track_data=True)
# Optionally: env = DummyVecEnv([lambda: env])


env_sc= MinMaxWrapper(env , clip_obs=True, clip_rew=True)


env_sc = Monitor(env_sc, filename="./sac_logs/monitor.csv")
env_sc = DummyVecEnv([lambda: env_sc])  #

pd.DataFrame(env.history)



model = SAC("MlpPolicy", env_sc, verbose=1)


import time

t1=time.time()
model.learn(total_timesteps=5000)
t2=time.time()
print("Training time (minutes):", (t2 - t1) / 60)
# Save or test the model
# Install stable-baselines3 using pip
# pip install stable-baselines3[extra]
#! pip install stable-baselines3


test_env= RetailerOrdersEnv(time_horizon=47, track_data=True)
test_env_sc= MinMaxWrapper(test_env , clip_obs=True, clip_rew=True)

obs, _ = test_env_sc.reset()

for step in range(47):
    action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions
    obs, reward, done, truncated, info = test_env_sc.step(action)
    if done:
        break

# View raw data collected in env
import pandas as pd
df = pd.DataFrame(test_env_sc.env.history)


import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

log_dir = "./sac_logs/"
x, y = ts2xy(load_results(log_dir), 'timesteps')

plt.figure(figsize=(10, 4))
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("SAC Training - Episode Reward Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()