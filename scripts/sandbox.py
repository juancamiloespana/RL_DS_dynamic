from RL_DS.envs.retailer_gym import RetailerOrdersEnv
from RL_DS_dynamic.RL_DS.utils.norm import ObsRewardNormalizeWrapper 
import gymnasium
import pandas as pd
import numpy as np



env= RetailerOrdersEnv(time_horizon=50, track_data=True)
env2= ObsRewardNormalizeWrapper(env)
env3=RetailerOrdersEnv(time_horizon=50, track_data=True)

env3.step(100)
env2.step(100)


pd.DataFrame(env3.history)

pd.DataFrame(env2.env.history)

env2.

