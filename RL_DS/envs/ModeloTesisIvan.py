import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RetailerOrders:

    # Método que inicializa todas las variables y parámetros del modelo
    def __init__(self):
        """
        Initialize parameters with default parameters.
        :param initial_channel_demand
        :param retailer_order_delay_time
        :param time_adjust_backlog
        :param target_delivery_delay
        :param normal_capacity
        :param supply_gap_cost_coefficient
        :param order_cost_coefficent
        """
        self.initial_channel_demand = 100.0 # Orders / weeks
        self.retailer_order_delay_time = 3 # weeks
        self.time_adjust_backlog = 3.0 # weeks
        self.target_delivery_delay = 10.0 # weeks
        self.normal_capacity = 100.0 # Orders / weeks
        self.supply_gap_cost_coefficient = 0.002 # USD / (weeks * orders^2)
        self.order_cost_coefficient = 0.001 # USD / orders
        """
        Initialize levels
        : initial level initial_backlog
        : initial level initial_capacity
        : initial level initial_cumulative_shipment
        : initial level initial_cumulative_customer_demand
        : initial level initial_Retailer_total_cost
        """
        self.initial_backlog = 1000.0 # orders
        self.initial_capacity = 100.0 # orders / weeks
        self.initial_cumulative_shipment = 100.0 # orders
        self.initial_cumulative_customer_demand = 100.0 # orders
        self.initial_Retailer_total_cost = 0 # USD

        """
        Levels definition
        :  level backlog
        :  level capacity
        :  level cumulative_shipment
        :  level cumulative_customer_demand
        :  level Retailer_total_cost
        """
        self.backlog = self.initial_backlog
        self.capacity = self.initial_capacity
        self.cumulative_shipment = self.initial_cumulative_shipment
        self.cumulative_customer_demand = self.initial_cumulative_customer_demand
        self.retailer_total_cost = self.initial_Retailer_total_cost

        """
        Auxiliaries variables definition
        : auxiliary customer_demand
        : auxiliary retailer_order
        : flow perceived_retailer_order
        : flow shipment_to_retailer
        : flow change_in_capacity
        : auxiliary desired_shipment
        : auxiliary delivery_delay
        : flow shipments
        : flow orders
        : auxiliary supply_gap
        : auxiliary supply_gap_costs
        : retailer_costs
        : flow costs
        """
        self.customer_demand = 0.0 # orders / weeks
        self.retailer_order = 0.0 # orders / weeks
        self.perceived_retailer_order = 0.0 # orders / weeks
        self.shipment_to_retailer = 0.0 # orders / weeks
        self.change_in_capacity = 0.0 # orders / weeks^2
        self.desired_shipment = 0.0 # orders / weeks
        self.delivery_delay = 0.0 # weeks
        self.shipments = 0.0 # orders / weeks
        self.orders = 0.0 # orders / weeks
        self.supply_gap = 0.0 # orders
        self.supply_gap_costs = 0.0 # usd / weeks
        self.retailer_costs = 0.0 # usd / weeks
        self.costs = 0.0 # usd / weeks
        
        """
        Simulation time and dt definition
        : simulation time 
        : dt
        : step
        """
        self.simulation_time = 36 # weeks
        self.dt = 1.0 # weeks
        self.step_sim = 0 # weeks
        
        """
        Delay ppl 
        : delay_vector
        """
        self.delay_vector = [0.0] * self.simulation_time

        """
        Action vector
        : action vector
        """
        self.action_list = [0.0]* self.simulation_time
        
    # Método de la función pulso
    def pulse(self,step, pulse_start =3, height = 20,pulse_end = 36):
        if(step >= pulse_start and step <= pulse_end):
            return height
        else:
            return 0
    
    # Método de la función delayppl
    def Delayppl(self, step ,variable, delay, initial_condition):
        # Vector que guarda los retardos
        self.delay_vector[step] = variable
        if(step >= delay):
            return self.delay_vector[step-delay]
        else:
            return initial_condition

    # Método que corre el modelo
    def run(self, action_decisions):

        self.action_list = action_decisions
        self.history = []
        for i in range(self.simulation_time):
            #print(i)
            # Niveles - Condiciones iniciales y ecuaciones diferenciales bajo método Euler
            if i == 0:
                # Definición de las condiciones iniciales de los niveles
                self.backlog = self.initial_backlog
                self.capacity = self.initial_capacity
                self.cumulative_shipment = self.initial_cumulative_shipment
                self.cumulative_customer_demand = self.initial_cumulative_customer_demand
                self.retailer_total_cost = self.initial_Retailer_total_cost               
            else:
                # Definición de las ecuaciones diferenciales bajo método de Euler
                self.backlog += (self.perceived_retailer_order-self.shipment_to_retailer)*self.dt
                self.capacity += (self.change_in_capacity)*self.dt
                self.cumulative_shipment += (self.shipments)*self.dt
                self.cumulative_customer_demand += (self.customer_demand)*self.dt
                self.retailer_total_cost += (self.costs)*self.dt

            # Variables auxiliares y parámetros    
            self.customer_demand = self.initial_channel_demand + self.pulse(i)
            self.retailer_order = self.action_list[i]
            self.perceived_retailer_order = self.Delayppl(i,self.retailer_order,self.retailer_order_delay_time,self.initial_capacity)
            self.shipment_to_retailer = self.capacity
            self.desired_shipment = self.backlog/self.target_delivery_delay
            self.change_in_capacity = (self.desired_shipment-self.capacity)/self.time_adjust_backlog
            self.delivery_delay = self.target_delivery_delay*(1+max(0, (self.desired_shipment-self.capacity)/self.normal_capacity))
            self.shipments = self.shipment_to_retailer
            self.orders = self.customer_demand
            self.supply_gap = self.cumulative_customer_demand-self.cumulative_shipment
            self.supply_gap_costs = self.supply_gap_cost_coefficient*(self.supply_gap**2) 
            self.retailer_costs = self.order_cost_coefficient*(self.retailer_order**2)
            self.costs = self.supply_gap_costs+self.retailer_costs
            
            
            self.history.append({
                "step": i,
                "demand": float(self.customer_demand),
                "action": float(self.action_list[i]),
                "backlog": float(self.backlog),
                "capacity": float(self.capacity),
                "change_in_capacity": float(self.change_in_capacity),
                "shipment_to_retailer": float(self.shipment_to_retailer),
                "desired_shipment": float(self.desired_shipment),
                "cumulative_demand": float(self.cumulative_customer_demand),
                "cumulative_shipment": float(self.cumulative_shipment),           
                #"supply_gap": float(self.supply_gap),
                "order_cost": float(self.retailer_costs),
                "supply_gap_cost": float(self.supply_gap_costs),
                "step_cost": float(self.costs),
                "cumulative_cost": float(self.retailer_total_cost)
            })

            
        #print(f' iteration {i} total cost {self.retailer_total_cost}')

        return self.retailer_total_cost


# actions=[30]*36
# actions=[268, 203, 177, 138, 128, 116,  80,  66,  98, 105, 118, 111, 122, 164,116, 139, 123, 114, 151, 101, 100,  99,  90,  99,  57,  69,  34,  37,
#   38,  55,  56,  47,  51,  48,  48,  78]
# len(actions)
# env= RetailerOrders()
# env.run(actions)

# import pandas as pd
# pd.DataFrame(env.history)

# num_samples = 1000   # Number of samples per iteration
# num_iterations = 500 # Number of iterations
# Model  = RetailerOrders()

# Mu = np.random.uniform(50, 400, 36)
# St = np.random.uniform(50, 50, 36)
# print(f' Mu inicial = {Mu}')
# #print(f' Std inicial = {St}')

# costos_optim = [0] * 10
# index_cost = 0 

# print(f'')
# for i in range(num_iterations):

#     Costs = [0.0] * num_samples
#     matrix = np.zeros((36, num_samples))

#     for j in range(num_samples):
#         random_numbers = np.abs(np.random.normal(Mu, St))
#         matrix[:,j] = random_numbers
#         Costs[j] = Model.run(random_numbers)
#         #print(f'Cost {j} = {Costs[j]}') 

#     sorted_indices = [i for i, _ in sorted(enumerate(Costs), key=lambda x: x[1])]
#     #print("")
#     #print(sorted_indices)
#     #print("")
#     #print(f'matrix {matrix}')

#     sorted_matrix = matrix[:, sorted_indices]
#     #print("")
#     #print(f'sorted matrix {sorted_matrix}')

#     selected_columns = sorted_matrix[:, :20]
#     #print("")
#     #print(f'selected columns {selected_columns}')

#     Mu = np.mean(selected_columns, axis=1)
#     #St = np.std(selected_columns, axis=1)

#     if i % 100 == 0:  # Print message every 10 iterations
#         costos_optim[index_cost] = min(Costs)
#         print(f'Iteration {i}, lowest cost = {costos_optim[index_cost]}')
#         index_cost+=1

# print(f'')
# #print(f' mu = {Mu}')
# Mu_real = np.round(Mu)
# print(f' mu = {Mu_real}')
# print(f'')
# prueba = RetailerOrders()
# print(f'Costo final = {prueba.run(Mu)}')

# plt.plot(costos_optim)
# plt.show()

