import gymnasium as gym
import numpy as np

class MinMaxWrapper(gym.Wrapper):
    """
    A wrapper that applies fixed min-max scaling to BOTH observations and rewards.
    
    For each observation dimension i in obs_shape:
        obs[i]_scaled = (obs[i] - obs_min[i]) / (obs_max[i] - obs_min[i])
        optionally clipped to [0, 1] if clip_obs=True

    For the reward:
        rew_scaled = (reward - rew_min) / (rew_max - rew_min)
        optionally clipped to [0, 1] if clip_rew=True
    """

    def __init__(
        self,
        env,
        obs_min_vals=[0,0,0,0,0,0,0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0],
        obs_max_vals=[5000,500,5000,10,5,5,100, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,500 ],
        rew_min_val=-5000000,
        rew_max_val=0,
        clip_obs=True,
        clip_rew=True
    ):
        """
        :param env: The original (unwrapped) environment
        :param obs_min_vals: 1D array or list of min values for each obs dimension
        :param obs_max_vals: 1D array or list of max values for each obs dimension
        :param rew_min_val: scalar minimum for rewards
        :param rew_max_val: scalar maximum for rewards
        :param clip_obs: whether to clip normalized obs to [0, 1]
        :param clip_rew: whether to clip normalized rew to [0, 1]
        """
        super().__init__(env)

        self.obs_min_vals = np.array(obs_min_vals, dtype=np.float32)
        self.obs_max_vals = np.array(obs_max_vals, dtype=np.float32)
        self.rew_min_val = float(rew_min_val)
        self.rew_max_val = float(rew_max_val)
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew

        # Validate shape compatibility for observations
        orig_obs_shape = self.env.observation_space.shape
        if self.obs_min_vals.shape != orig_obs_shape or self.obs_max_vals.shape != orig_obs_shape:
            raise ValueError(f"obs_min_vals and obs_max_vals must match observation shape {orig_obs_shape}.")

        # Create a new Box space for normalized obs in [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=orig_obs_shape,
            dtype=np.float32
        )

        # Reward is a scalar or sometimes a 1D shape=() in Gym.
        # We'll assume it's a scalar that we scale to [0..1] if clip_rew=True.

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        obs = self._normalize_obs(obs)
        reward = self._normalize_reward(reward)

        return obs, reward, done, truncated, info

    def _normalize_obs(self, obs):
        # Per-dimension scaling
        norm_obs = (obs - self.obs_min_vals) / (self.obs_max_vals - self.obs_min_vals + 1e-8)

        if self.clip_obs:
            norm_obs = np.clip(norm_obs, 0.0, 1.0)

        return norm_obs.astype(np.float32)

    def _normalize_reward(self, reward):
        # Scalar scaling
        norm_rew = (reward - self.rew_min_val) / (self.rew_max_val - self.rew_min_val + 1e-8)

        if self.clip_rew:
            norm_rew = float(np.clip(norm_rew, 0.0, -1.0))
        else:
            norm_rew = float(norm_rew)

        return norm_rew

# obs = np.concatenate((np.array([
#     self.backlog,
#     self.capacity,
#     supply_gap,
#     float(self.target_delivery_delay),
#     float(self.retailer_order_delay_time), 
#     float(self.time_adjust_backlog ),
#     time_remaining], dtype=np.float32), demand_profile))

# obs_min_vals=[0,0,0,0,0,0,0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# obs_max_vals=[5000,500,5000,10,5,5,100, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,500]
# rew_min_val=-5000000,
# rew_max_val=0,



### uso 

from RL_DS.envs.retailer_gym import RetailerOrdersEnv

env_raw = RetailerOrdersEnv(time_horizon=36, track_data=True)
env_scaled = MinMaxWrapper(
    env_raw,
    obs_min_vals,
    obs_max_vals,
    rew_min_val,
    rew_max_val,
    clip_obs=True,
    clip_rew=True
)

obs, info = env_scaled.reset()
print("Scaled initial obs:", obs)  # Should be in [0..1] range per dimension
env_raw.history
env_scaled.reward
action = env_scaled.action_space.sample()
obs, rew, done, truncated, info = env_scaled.step(0)
print( rew)

import pandas as pd
pd.DataFrame(env_raw.history)