import numpy as np
import gymnasium as gym
from RL_DS.envs.retailer_gym import RetailerOrdersEnv
from RL_DS.utils.norm import ObsRewNormWrapper 
# Assuming you have these classes/files:
# - RetailerOrdersEnv (your environment)
# - ObsRewardNormalizeWrapper (your normalization wrapper)

def test_wrapper():
    # 1) Create an unwrapped environment
    env_raw = RetailerOrdersEnv(time_horizon=20000, track_data=True)
    # 2) Create a wrapped environment
    env_wrapped = ObsRewNormWrapper(RetailerOrdersEnv(time_horizon=2000, track_data=True))

    obs_raw, _ = env_raw.reset()
    obs_wrapped, _ = env_wrapped.reset()

    raw_obses = []
    wrapped_obses = []
    raw_rewards = []
    wrapped_rewards = []

    done_raw = False
    done_wrapped = False

    steps = 2000
    for _ in range(steps):
        # Use the same random action in both envs
        action = env_raw.action_space.sample()

        if not done_raw:
            obs_r, rew_r, done_raw, _, _ = env_raw.step(action)
            raw_obses.append(obs_r)
            raw_rewards.append(rew_r)

        if not done_wrapped:
            obs_w, rew_w, done_wrapped, _, _ = env_wrapped.step(action)
            wrapped_obses.append(obs_w)
            wrapped_rewards.append(rew_w)

    raw_obses = np.array(raw_obses)
    wrapped_obses = np.array(wrapped_obses)
    raw_rewards = np.array(raw_rewards)
    wrapped_rewards = np.array(wrapped_rewards)

    print("Raw Observations:")
    print("  Mean:", raw_obses.mean(axis=0), "Std Dev:", raw_obses.std(axis=0))
    print("Wrapped Observations:")
    print("  Mean:", wrapped_obses.mean(axis=0), "Std Dev:", wrapped_obses.std(axis=0))

    print("Raw Rewards:")
    print("  Mean:", raw_rewards.mean(), "Std Dev:", raw_rewards.std())
    print("Wrapped Rewards:")
    print("  Mean:", wrapped_rewards.mean(), "Std Dev:", wrapped_rewards.std())

if __name__ == "__main__":
    test_wrapper()

import pandas as pd
pd.DataFrame(env_raw.history)
pd.DataFrame(env_wrapped.env.history)

raw_rewards = np.array(raw_rewards)

mean_raw_rewards = raw_rewards.mean()
std_raw_rewards = raw_rewards.std()

normalized_rewards = (raw_rewards - mean_raw_rewards) / std_raw_rewards

print("Normalized Rewards:")
print("  Mean:", normalized_rewards.mean(), "Std Dev:", normalized_rewards.std())

normalized_rewards==wrapped_rewards


x=raw_rewards
  
x = np.asarray(x, dtype=np.float64)
batch_count = x.shape[0] if len(x.shape) > 1 else 1
batch_mean = np.mean(x, axis=0)
batch_var = np.var(x, axis=0)

std_raw_rewards**2

new_count = self.count + batch_count
delta = batch_mean - self.mean
# total_var accumulates the previous var + new batch var + correction factor
total_var = (self.var * self.count +
                batch_var * batch_count +
                delta**2 * (self.count * batch_count / new_count))

# Update the mean and variance
self.mean += delta * (batch_count / new_count)
self.var = total_var / new_count
self.count = new_count