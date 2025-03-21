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
        self.simulation_time = 47 # weeks
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
    def pulse(self,step, pulse_start =3, height = 20):
      
        if(step >= pulse_start and step <= self.simulation_time):
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

############# pruebas del entorno #######
 
actions_de=[257.88594829, 209.57960392, 170.28192868, 140.48524976,
       119.74589523, 106.94422777, 100.59021471,  98.9578117 ,
       100.57384745, 104.00321529, 108.17754893, 112.30625588,
       115.84292628, 118.57892995, 120.43359765, 121.46822124,
       121.90404634, 121.93688061, 121.74003065, 121.60987643,
       121.69458526, 122.0743363 , 122.85516796, 123.9553942 ,
       125.26283814, 126.52591904, 127.38528922, 127.36169008,
       125.95921366, 122.54426307, 116.60612778, 107.75949482,
        95.82931333,  81.036542  ,  64.10373362,  46.17529297,
        28.98076765,  14.49374935,   4.49882298,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   8.56208228, 272.24740514]

# actions_fix=[100]*47

# actions_CE=[259.68109179, 210.91152195, 171.35084108, 137.64783849,
#        116.81336704, 106.90350338,  99.21243418, 100.90581488,
#        101.57465841, 103.55455529, 111.8241507 , 108.79596831,
#        119.52606938, 115.22771514, 119.91098701, 122.38543703,
#        124.69295067, 117.84640877, 119.10740143, 124.29103364,
#        121.24917079, 125.6030072 , 120.21028493, 118.79460864,
#        122.11987681, 127.64285086, 126.83176275, 128.09689893,
#        124.57254739, 129.66738289, 123.09179014, 118.53563538,
#        105.19362769,  94.02391053,  84.70358147,  58.94126073,
#         48.46636759,  25.63990819,  11.49118603,  14.05944294,
#          9.08035835,   9.46166999,   8.43878904,   9.23707975,
#         12.55709491,   8.60496668, 213.36166669]


# len(actions_CE)
# env= RetailerOrders()
# env.run(actions_de)
# env.retailer_total_cost
# import pandas as pd
# pd.DataFrame(env.history)


