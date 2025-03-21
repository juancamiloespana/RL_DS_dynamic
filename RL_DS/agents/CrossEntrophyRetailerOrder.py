from RL_DS.envs.ModeloTesisIvan import RetailerOrders
import numpy as np
import csv
import pandas as pd


class CrossEntropyMethod:
    def __init__(self, model, num_samples=2000, num_iterations=500, elite_columns=20):
        self.model = model  # System dynamics model
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.elite_columns = elite_columns
        self.simulation_time = model.simulation_time
        
        # Initialize mean and standard deviation
        self.Mu = np.random.uniform(50, 400, self.simulation_time)
        self.St = [10] * self.simulation_time
        
        # Best actions tracking
        self.best_actions = [0] * self.simulation_time
        self.best_cost = model.run(self.Mu)
    
    def optimize(self):
        """Runs the Cross-Entropy optimization method."""
        for i in range(self.num_iterations):
            costs = [0.0] * self.num_samples
            matrix = np.zeros((self.simulation_time, self.num_samples))

            # Generate samples
            for j in range(self.num_samples):
                random_numbers = np.abs(np.random.normal(self.Mu, self.St))
                matrix[:, j] = random_numbers
                costs[j] = self.model.run(random_numbers)

            # Sort by cost
            sorted_indices = sorted(range(len(costs)), key=lambda x: costs[x])
            sorted_matrix = matrix[:, sorted_indices]
            selected_columns = sorted_matrix[:, :self.elite_columns]
            
            # Update mean
            self.Mu = np.mean(selected_columns, axis=1)

            # Track best cost and actions
            cost = self.model.run(self.Mu)
            if cost < self.best_cost:
                self.best_actions = self.Mu
                self.best_cost = cost

            if i % 50 == 0:
                print(f"Iteration {i} lowest cost = {self.best_cost}")

        print("\nBest Actions:", self.best_actions)
        print(f"Total Cost: {self.best_cost}\n")
    
    def save_results(self, filename="actions.csv"):
        """Saves the best actions to a CSV file."""
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Example Input", "Example Output"])
            for index, num in enumerate(self.best_actions):
                writer.writerow([index, num])
        print(f"CSV file '{filename}' created successfully!")

# Example usage:
Model = RetailerOrders()
cem = CrossEntropyMethod(Model)
cem.optimize()
# cem.save_results()


# pd.DataFrame(Model.history)

# cem.best_actions

# Model.run(cem.best_actions)
# Model.history