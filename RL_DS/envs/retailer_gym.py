import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import pandas as pd

class RetailerOrdersEnv(gym.Env):
    """
    Gymnasium environment for simulating retailer orders and supply chain dynamics.
    Tracks step-by-step information if track_data=True.
    """

    metadata = {"render.modes": []}

    def __init__(self, time_horizon=50, track_data=False):
        super(RetailerOrdersEnv, self).__init__()

        # Store parameters
        self.time_horizon = time_horizon
        self.track_data = track_data

        # Environment constants
        self.initial_channel_demand = 100      # Orders per week
        self.retailer_order_delay_time = 3     # Weeks
        self.time_adjust_backlog = 3           # Weeks
        self.target_delivery_delay = 10        # Weeks
        self.normal_capacity = 100             # Orders per week
        self.supply_gap_cost_coefficient = 0.002   # USD / (weeks * orders^2)
        self.order_cost_coefficient = 0.001         # USD / orders

        # Initial Conditions
        self.initial_backlog = 1000
        self.initial_capacity = 100
        self.initial_cumulative_shipment = 0
        self.initial_cumulative_customer_demand = 0
        self.initial_retailer_total_cost = 0

        # Derived variable
        self.desired_shipment = self.initial_backlog / self.target_delivery_delay

        # Gym Spaces
        obs_size= self.time_horizon + 7
        self.action_space = spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_size,), dtype=np.float32
        )  # Example: [initial_channel_demand, backlog, target_delivery_delay, 0]

        # Setup Logger
        logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

        # Step Data Storage (only if track_data=True)
        self.history = [] if self.track_data else None

        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)

        # State variables
        self.backlog = self.initial_backlog
        self.capacity = self.initial_capacity
        self.cumulative_shipment = self.initial_cumulative_shipment
        self.cumulative_customer_demand = self.initial_cumulative_customer_demand
        self.retailer_total_cost = self.initial_retailer_total_cost

        # For the delayed action effect
        self.delay_vector = np.zeros(self.time_horizon)

        # Time-step counter
        self.current_step = 0

        # Clear history if tracking is enabled
        if self.track_data:
            self.history = []

        return self._get_observation(), {}

    def step(self, action):
        """Advances the environment by one time step based on the action."""
        # Clip the action within [0, 500]
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]

        # Determine this step's customer demand
        customer_demand = self.initial_channel_demand + self._pulse(self.current_step)

        # Update capacity
        change_in_capacity = (self.desired_shipment - self.capacity) / self.time_adjust_backlog
        self.capacity += change_in_capacity

        # Update desired shipment based on backlog
        self.desired_shipment = self.backlog / self.target_delivery_delay

        # Determine perceived order (delayed by retailer_order_delay_time)
        perceived_retailer_order = self._delayed_signal(
            self.current_step, action, self.retailer_order_delay_time, self.initial_capacity
        )

        # Shipment to the retailer is limited by capacity or desired_shipment
        shipment_to_retailer = min(self.capacity, self.desired_shipment)

        # Update backlog
        self.backlog += (perceived_retailer_order - shipment_to_retailer)

        # Update cumulative metrics
        self.cumulative_shipment += shipment_to_retailer
        self.cumulative_customer_demand += customer_demand

        # Calculate costs
        supply_gap = self.cumulative_customer_demand - self.cumulative_shipment
        supply_gap_costs = self.supply_gap_cost_coefficient * (supply_gap ** 2)
        retailer_costs = self.order_cost_coefficient * (action ** 2)
        step_cost = supply_gap_costs + retailer_costs

        # For RL training, you'd typically use incremental reward:
        # reward = -step_cost
        # But for now, the code returns the *cumulative* cost to maintain original logic.
        self.retailer_total_cost += step_cost
        reward = -step_cost  # Original approach

        # Check termination
        self.current_step += 1
        done = self.current_step >= self.time_horizon

        # (Optional) Log step data if track_data is True
        if self.track_data:
            self.history.append({
                "step": self.current_step,
                "action": float(action),
                "demand": float(customer_demand),
                "capacity": float(self.capacity),
                "desired_shipment": float(self.desired_shipment),
                "change_in_capacity": float(change_in_capacity),
                "shipment_to_retailer": float(shipment_to_retailer),
                "backlog": float(self.backlog),
                "supply_gap": float(supply_gap),
                "supply_gap_cost": float(supply_gap_costs),
                "order_cost": float(retailer_costs),
                "step_cost": float(step_cost),
                "reward": float(reward),
                "cumulative_cost": float(self.retailer_total_cost)
            })

        return self._get_observation(), reward, done, False, {}

    def _pulse(self, step, start=2, height=20):
        """Returns a pulse signal within a specific range."""
        return height if start <= step <= self.time_horizon else 0

    def _delayed_signal(self, step, value, delay, initial_value):
        """Returns a delayed version of the action using a simple delay buffer."""
        self.delay_vector[step] = value
        if step >= delay:
            return self.delay_vector[step - delay]
        else:
            return initial_value

    def _get_observation(self):
        """
        Returns the current state observation.
    
        """
        time_remaining = self.time_horizon - self.current_step
        supply_gap = self.cumulative_customer_demand - self.cumulative_shipment
        demand_profile = np.zeros(self.time_horizon, dtype=np.float32)
        
        for step_idx in range(self.time_horizon):
            demand_profile[step_idx] = self.initial_channel_demand + self._pulse(step_idx)

        obs = np.concatenate((np.array([
            self.backlog,
            self.capacity,
            supply_gap,
            float(self.target_delivery_delay),
            float(self.retailer_order_delay_time), 
            float(self.time_adjust_backlog ),
            time_remaining], dtype=np.float32), demand_profile))


        return obs

    def close(self):
        """Placeholder for environment closing logic if needed."""
        pass

# env= RetailerOrdersEnv(time_horizon=35, track_data=True)

# obs, *_=env.step(150)
# env.history

# pd.DataFrame(env.history)

