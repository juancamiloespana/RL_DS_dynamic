import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys



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
        self.initial_channel_demand = 100 # Orders / weeks
        self.retailer_order_delay_time = 3 # weeks
        self.time_adjust_backlog = 3 # weeks
        self.target_delivery_delay = 10 # weeks
        self.normal_capacity = 100 # Orders / weeks
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
        self.initial_backlog = 1000 # orders
        self.initial_capacity = 100 # orders / weeks
        self.initial_cumulative_shipment = 100 # orders
        self.initial_cumulative_customer_demand = 100 # orders
        self.initial_Retailer_total_cost = 0 # USD

        """
        Levels definition
        :  level backlog
        :  level capacity
        :  level cumulative_shipment
        :  level cumulative_customer_demand
        :  level Retailer_total_cost
        """
        self.backlog = [self.initial_backlog]
        self.capacity = [self.initial_capacity]
        self.cumulative_shipment = [self.initial_cumulative_shipment]
        self.cumulative_customer_demand = [self.initial_cumulative_customer_demand]
        self.retailer_total_cost = [self.initial_Retailer_total_cost]

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
        self.customer_demand = 0 # orders / weeks
        self.retailer_order = 0 # orders / weeks
        self.perceived_retailer_order = 0 # orders / weeks
        self.shipment_to_retailer = 0 # orders / weeks
        self.change_in_capacity = 0 # orders / weeks^2
        self.desired_shipment = 0 # orders / weeks
        self.delivery_delay = 0 # weeks
        self.shipments = 0 # orders / weeks
        self.orders = 0 # orders / weeks
        self.supply_gap = 0 # orders
        self.supply_gap_costs = 0 # usd / weeks
        self.retailer_costs = 0 # usd / weeks
        self.costs = 0 # usd / weeks

        """
        Simulation time and dt definition
        : simulation time 
        : dt
        : step
        """
        self.simulation_time = 35 # weeks
        self.dt = 1 # weeks
        self.step_sim = 0 # weeks
        """
        Delay ppl 
        : delay_vector
        """
        self.delay_vector = [0] * self.simulation_time
        """
        Cross Entrophy variables definition
        : Actions actions
        : Observations obs
        : Observations size obs_size
        """
        # Acciones
        self.actions = [40,30,20,10,0,-10,-20,-30,-40]
        # Observaciones 
        self.obs = [] * 4 # Numero de observaciones
        """ 
        Observations:
        Customer_demand, Backlog, Target_delivery_delay and Delivery_delay
        """
        self.obs_size = 4

        """"Results initialization"""
        self.Results_step = (self.obs, -1 * self.initial_Retailer_total_cost, False)

        

    # Metodo que resetea las variables
    def reset(self):
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
        self.initial_channel_demand = 100 # Orders / weeks
        self.retailer_order_delay_time = 3 # weeks
        self.time_adjust_backlog = 3 # weeks
        self.target_delivery_delay = 10 # weeks
        self.normal_capacity = 100 # Orders / weeks
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
        self.initial_backlog = 1000 # orders
        self.initial_capacity = 100 # orders / weeks
        self.initial_cumulative_shipment = 100 # orders
        self.initial_cumulative_customer_demand = 100 # orders
        self.initial_Retailer_total_cost = 0 # USD

        """
        Levels definition
        :  level backlog
        :  level capacity
        :  level cumulative_shipment
        :  level cumulative_customer_demand
        :  level Retailer_total_cost
        """
        self.backlog = [self.initial_backlog]
        self.capacity = [self.initial_capacity]
        self.cumulative_shipment = [self.initial_cumulative_shipment]
        self.cumulative_customer_demand = [self.initial_cumulative_customer_demand]
        self.retailer_total_cost = [self.initial_Retailer_total_cost]

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
        self.customer_demand = 0 # orders / weeks
        self.retailer_order = 0 # orders / weeks
        self.perceived_retailer_order = 0 # orders / weeks
        self.shipment_to_retailer = 0 # orders / weeks
        self.change_in_capacity = 0 # orders / weeks^2
        self.desired_shipment = 0 # orders / weeks
        self.delivery_delay = 0 # weeks
        self.shipments = 0 # orders / weeks
        self.orders = 0 # orders / weeks
        self.supply_gap = 0 # orders
        self.supply_gap_costs = 0 # usd / weeks
        self.retailer_costs = 0 # usd / weeks
        self.costs = 0 # usd / weeks

        """
        Simulation time and dt definition
        """
        self.step_sim = 0 # weeks
        """
        Delay ppl 
        : delay_vector
        """
        self.delay_vector = [0] * self.simulation_time

        """"Results initialization"""
        self.Results_step = (self.obs, -1 * self.initial_Retailer_total_cost, False)

        # Se genera la primera observacion
        self.obs = [self.customer_demand,self.backlog,self.target_delivery_delay,self.delivery_delay]
        return self.obs
        
    # Método que indica cuando parar
    def is_done(self):
        if self.step_sim == self.simulation_time:
            return True
        else:
            return False
        
    # Metodo para determinar los indices de las acciones
    def action_index(self, action_number: int):      
    
        if 0 <= action_number < len(self.actions):
            decisiones = self.actions[action_number]  # Ensure it's a valid list
        else:
            print("Invalid action")
            return []
    
        return decisiones
       
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
        if(step > delay):
            return self.delay_vector[step-delay]
        else:
            return initial_condition

    # Método que corre el modelo
    def run(self):

        for i in range(self.simulation_time):

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
                self.cumulative_customer_demand += (self.orders)*self.dt
                self.retailer_total_cost += (self.costs)*self.dt

            # Variables auxiliares y parámetros    
            self.customer_demand = self.initial_channel_demand + self.pulse(i)
            self.retailer_order = self.customer_demand
            self.perceived_retailer_order = self.Delayppl(i,self.retailer_order,self.retailer_order_delay_time,self.initial_capacity)
            self.shipment_to_retailer = self.capacity
            self.desired_shipment = self.backlog/self.target_delivery_delay
            self.change_in_capacity = (self.desired_shipment-self.capacity)/self.time_adjust_backlog
            self.delivery_delay = self.target_delivery_delay*(1+max(0, (self.desired_shipment-self.capacity)/self.normal_capacity))
            self.shipments = self.shipment_to_retailer
            self.orders = self.retailer_order
            self.supply_gap = self.cumulative_customer_demand-self.cumulative_shipment
            self.supply_gap_costs = self.supply_gap_cost_coefficient*(self.supply_gap**2) 
            self.retailer_costs = self.order_cost_coefficient*(self.retailer_order**2)
            self.costs = self.supply_gap_costs+self.retailer_costs
            

    # Metodo que corre el modelo paso a paso
    def step(self, action : int):

        # Niveles - Condiciones iniciales y ecuaciones diferenciales bajo método Euler
        if self.step_sim == 0:
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
            self.cumulative_customer_demand += (self.orders)*self.dt
            self.retailer_total_cost += (self.costs)*self.dt

        # Variables auxiliares y parámetros    
        self.customer_demand = self.initial_channel_demand + self.pulse(self.step_sim)
        """En retailer_order sumo la acción del modelo"""
        self.retailer_order = 100 + self.action_index(action)
        self.perceived_retailer_order = self.Delayppl(self.step_sim,self.retailer_order,self.retailer_order_delay_time,self.initial_capacity)
        self.shipment_to_retailer = self.capacity
        self.desired_shipment = self.backlog/self.target_delivery_delay
        self.change_in_capacity = (self.desired_shipment-self.capacity)/self.time_adjust_backlog
        self.delivery_delay = self.target_delivery_delay*(1+max(0, (self.desired_shipment-self.capacity)/self.normal_capacity))
        self.shipments = self.shipment_to_retailer
        self.orders = self.retailer_order
        self.supply_gap = self.cumulative_shipment
        self.supply_gap_costs = self.supply_gap_cost_coefficient*(self.supply_gap**2) 
        self.retailer_costs = self.order_cost_coefficient*(self.retailer_order**2)
        self.costs = self.supply_gap_costs+self.retailer_costs

        # Observaciones
        self.obs = [self.customer_demand,self.backlog,self.target_delivery_delay,self.delivery_delay]
        self.step_sim += 1
        self.Results_step = (self.obs,-1*self.retailer_total_cost,self.is_done())
        
        return self.Results_step     

    # Método que grafica las variables de interés
    def plot_simulation(self):
        time_points = np.linspace(0, self.simulation_time, len(self.capacity))
        plt.plot(time_points, self.Retailer_total_cost, label='Retailer total cost')
        plt.plot(time_points, self.capacity, label='Capacity')
        plt.plot(time_points, self.backlog, label='Backlog')
        plt.xlabel('Time')
        plt.ylabel('Orders / weeks')
        plt.title('Retailer orders')
        plt.legend()
        plt.show()      
        

    
env.plot_simulation()