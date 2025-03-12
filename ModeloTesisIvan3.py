import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

class RetailerOrdersEnv(gym.Env):
    """Gymnasium environment for simulating retailer orders and supply chain dynamics."""

    metadata = {"render.modes": []}

    def __init__(self, time_horizon=50):
        super(RetailerOrdersEnv, self).__init__()

        # Parameters
        self.initial_channel_demand = 100  # Orders per week
        self.retailer_order_delay_time = 3  # Weeks
        self.time_adjust_backlog = 3  # Weeks
        self.target_delivery_delay = 10  # Weeks
        self.normal_capacity = 100  # Orders per week
        self.supply_gap_cost_coefficient = 0.002  # USD / (weeks * orders^2)
        self.order_cost_coefficient = 0.001  # USD / orders

        # Initial State
        self.initial_backlog = 1000
        self.initial_capacity = 100
        self.initial_cumulative_shipment = 0
        self.initial_cumulative_customer_demand = 0
        self.initial_retailer_total_cost = 0
        self.desired_shipment =   self.initial_backlog /self.target_delivery_delay # orders / weeks

        # Simulation Setup
        self.time_horizon = time_horizon
        self.dt = 1  # Simulation step size (weeks)
        self.current_step = 0

        # Action & Observation Spaces
        self.action_space = spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )  # Customer Demand, Backlog, Target Delivery Delay, Delivery Delay

        # Logging
        logging.basicConfig(level=logging.INFO)

        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)

        self.backlog = self.initial_backlog
        self.capacity = self.initial_capacity
        self.cumulative_shipment = self.initial_cumulative_shipment
        self.cumulative_customer_demand = self.initial_cumulative_customer_demand
        self.retailer_total_cost = self.initial_retailer_total_cost

        self.delay_vector = np.zeros(self.time_horizon)
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):
        """Advances the environment by one time step based on the action."""
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]

        customer_demand = self.initial_channel_demand + self._pulse(self.current_step)
        change_in_capacity = (self.desired_shipment - self.capacity) / self.time_adjust_backlog
        self.capacity += change_in_capacity * self.dt
        self.desired_shipment = self.backlog / self.target_delivery_delay
        
        perceived_retailer_order = self._delayed_signal(
            self.current_step, action, self.retailer_order_delay_time, self.initial_capacity
        )
        shipment_to_retailer = min(self.capacity, desired_shipment)
        delivery_delay = self.target_delivery_delay * (
            1 + max(0, (desired_shipment - self.capacity) / self.normal_capacity)
        )

        self.backlog += (perceived_retailer_order - shipment_to_retailer) * self.dt
        self.cumulative_shipment += shipment_to_retailer * self.dt
        self.cumulative_customer_demand += customer_demand * self.dt
        supply_gap = self.cumulative_customer_demand - self.cumulative_shipment

        supply_gap_costs = self.supply_gap_cost_coefficient * (supply_gap**2)
        retailer_costs = self.order_cost_coefficient * (action**2)
        total_cost = supply_gap_costs + retailer_costs
        print(supply_gap_costs)
        print(retailer_costs)
        self.retailer_total_cost += total_cost * self.dt
        self.current_step += 1

        done = self.current_step >= self.time_horizon
        reward =  self.retailer_total_cost  # Minimizing costs as the goal

        logging.info(
            f"Step: {self.current_step}, Action: {action}, Backlog: {self.backlog:.2f}, "
            f"Delivery Delay: {delivery_delay:.2f}, Cost: {total_cost:.2f}"
        )

        return self._get_observation(), reward, done, False, {}

    def _pulse(self, step, start=2, height=20):
        """Returns a pulse signal within a specific range."""
        return height if start <= step <= self.time_horizon else 0

    def _delayed_signal(self, step, value, delay, initial_value):
        """Returns a delayed version of a variable using a simple delay vector."""
        self.delay_vector[step] = value
        return self.delay_vector[step - delay] if step >= delay else initial_value

    def _get_observation(self):
        """Returns the current state observation."""
        return np.array(
            [self.initial_channel_demand, self.backlog, self.target_delivery_delay, 0], dtype=np.float32
        )

    def close(self):
        """Placeholder for environment closing logic if needed."""
        pass


env= RetailerOrdersEnv(time_horizon=35)

env.step(150)

env.supply_gap_costs
env.retailer_total_cost
env.capacity