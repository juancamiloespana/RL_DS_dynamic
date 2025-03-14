from RL_DS.envs.retailer_gym import RetailerOrdersEnv
from utils.normalization import ObsRewardNormalizeWrapper 
import gym
import pandas as pd
import numpy as np
import os





env= RetailerOrdersEnv(time_horizon=50, track_data=True)

env2= ObsRewardNormalizeWrapper(env)