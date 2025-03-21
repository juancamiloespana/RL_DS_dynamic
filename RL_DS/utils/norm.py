import gymnasium as gym
import numpy as np

class RunningMeanStd:
    """
    Tracks mean, variance, and count for a data stream.
    Allows incremental updates as new data arrives.
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # to avoid division by zero

    def update(self, x):
        """
        Update the running mean and variance with a new batch of data `x`.
        x can be a single observation or a batch of observations.
        """
        x = np.asarray(x, dtype=np.float64)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

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

    def normalize(self, x, eps=1e-8):
        """
        Return the standardized value of x using the current mean and variance.
        """
        return (x - self.mean) / (np.sqrt(self.var) + eps)




class ObsRewNormWrapper(gym.Wrapper):
    """
    A gym wrapper that standardizes both observations and rewards
    using running mean and std tracking.
    """
    def __init__(self, env, obs_eps=1e-8, rew_eps=1e-8):
        super().__init__(env)
        # Setup observation normalization
        obs_shape = env.observation_space.shape
        self.rms_obs = RunningMeanStd(shape=obs_shape)
        self.obs_eps = obs_eps

        # Setup reward normalization
        # Rewards are scalars or vectors?
        # If it's a scalar reward, shape=()
        # If your environment returns a vector reward, use that shape
        self.rms_rew = RunningMeanStd(shape=())

        self.rew_eps = rew_eps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Update and normalize observation
        self.rms_obs.update(obs)
        obs = self.rms_obs.normalize(obs, self.obs_eps)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Normalize obs
        self.rms_obs.update(obs)
        obs = self.rms_obs.normalize(obs, self.obs_eps)
        
        # Normalize reward
        # For each step's scalar reward, update stats
        self.rms_rew.update([reward])  # pass as list or array
        reward = float(self.rms_rew.normalize([reward], self.rew_eps))

        return obs, reward, done, truncated, info
    


 