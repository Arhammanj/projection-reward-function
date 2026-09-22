#!/usr/bin/env python3
#
#  tcl_env.py
#  TCL environment for RL algorithms
#
# Author: Ehtisham Asghar

import random
import numpy as np
import pandas as pd
import os
import pathlib
from matplotlib import pyplot as plt
# Run only if Kernel is dying due to matplotlib plt command
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

root_dir = pathlib.Path(__file__).parent.resolve().parent
import gym
# Trying out if this works for others. from gym import spaces had some issues
import gym.spaces as spaces
import threading
import math
from openpyxl import load_workbook
# Default parameters for
# From Taha's code
# days range
DEFAULT_DAY0=0
DEFAULT_DAYN=300
# PV power generated in the microgrid
# DEFAULT_POWER_GENERATED = np.genfromtxt("Nim-PV.csv", delimiter=',', skip_header=0, usecols=[-1])

# Grid ToU prices
# DEFAULT_TOU_PRICE = np.genfromtxt("tou_price.csv", delimiter=',', skip_header=0, usecols=[-1])

# data link for pc simulation
# DEFAULT_TOU_PRICE = np.genfromtxt("c:\\RL-Git-Projects\\BMS-DQN\\datasets\\Prices.csv", delimiter=';', skip_header=1, usecols=[-1])

# Load real wholesale prices from datasets/Prices.csv
_csv_path = os.path.join(root_dir, 'datasets', 'Prices.csv')
if os.path.exists(_csv_path):
    DEFAULT_TOU_PRICE = np.genfromtxt(
        _csv_path, delimiter=';', skip_header=1, usecols=[1]
    ).astype(np.float32)
    # CSV prices are in pence/kWh; scale to £/kWh
    DEFAULT_TOU_PRICE = DEFAULT_TOU_PRICE / 100.0
else:
    raise FileNotFoundError(
        f"Price data not found at {_csv_path}. "
        "Place Prices.csv in the datasets/ folder."
    )

# DEFAULT_PEN_UNB = 50000

# Length of one episode (24 hrs)
DEFAULT_ITERATIONS = 24

# Loads params
# DEFAULT_BASE_LOAD = np.genfromtxt("Average_Load.csv", delimiter=',', skip_header=0, usecols=[-1])

# Battery characteristics (kwh)
DEFAULT_BAT_CAPACITY=400.0
DEFAULT_MAX_SOC = 0.8       # max soc 0.8
DEFAULT_MIN_SOC = 0.2       # min soc 0.2
DEFAULT_EFFICIENCY = 1       # charge/discharge efficiency = 1
DEFAULT_DEGRADATION = 0.001      # battery degradation coefficent = 0
DEFAULT_PEN_SOC = 500        # SOC penalty set to 50000

# Dg parameters
# DEFAULT_A = 0.0027
# DEFAULT_B = 0.02

# ── Projection reward hyper-parameters ───────────────────────────────────────
# These are the only new constants added on top of the paper baseline.
#
# PROJ_BETA — gap penalty weight.
#   Controls how hard the agent is penalised for choosing an action that the
#   battery physics had to override.  Higher = agent learns faster to stay
#   away from SOC walls, but too high can suppress exploration.
#   Recommended starting range: [0.5, 3.0]
DEFAULT_PROJ_BETA = 1.5

# PROJ_GAMMA — price alignment weight.
#   Rewards the agent for charging when price is low and discharging when
#   price is high.  Same economic intuition as the baseline but made explicit.
#   Recommended starting range: [0.1, 0.5]
DEFAULT_PROJ_GAMMA = 0.3
# ─────────────────────────────────────────────────────────────────────────────

MAX_R = 100

# Rendering lists
# LOADS_RENDER = []
BATTERY_RENDER = []
BATTERY_CH_DCH_RANDER = []
PRICE_RENDER = []
ENERGY_GRID_RENDER = []
# ENERGY_R_UNBALANCE_RENDER = []
GRID_PRICES_BUY_RENDER = []
GRID_PRICES_SELL_RENDER = []
ENERGY_GENERATED_RENDER = []
# TOTAL_CONSUMPTION_RENDER=[]
# DG_GEN_RENDER = []
OPERATION_COST = []
OP_CO_woDC = []

#ACTIONS = [[i, j] for i in range(0, 220, 20) for j in range(-80, 120, 40)]
# ACTIONS = [[i, j] for i in range(0, 220, 20) for j in range(-80, 100, 20)]
#ACTIONS = [[i, j] for i in range(0, 220, 20) for j in range(-80, 85, 5)]
#ACTIONS = [[i, j] for i in range(0, 205, 5) for j in range(-80, 85, 5)]

ACTIONS = [i for i in range(-80, 100, 20)]    # [-80, -60, -40....60, 80]

# ACTIONS = [i for i in range(-80, 90, 10)]

# class Battery():
#     '''simulate a simple battery here'''
#     def __init__(self,capacity, max_soc, min_soc, efficiency, degradation, penalty_soc):
#         self.capacity = capacity # max capacity 200 kwh
#         self.max_soc = max_soc # max soc 0.8
#         self.min_soc = min_soc # 0.2
#         self.efficiency = efficiency # charge and discharge efficiency 1.0
#         self.degradation = degradation # degradation cofficient set to 0
#         self.pen_soc = penalty_soc

#     def step(self,battery_action):
#         p_bat = battery_action
#         #print(f'SOC Before: {self.current_capacity}')
#         #print(f'Bat Action: {p_bat}') 
#         #print(f'max soc: {self.max_soc}')
#         #print(f'min soc: {self.min_soc}')
#         #print(f'cap: {self.capacity}')
#         updated_capacity = (self.current_capacity+(p_bat/self.capacity))
#         #print(f'Updated capacity: {updated_capacity}')
#         #print(f'current cap: {self.current_capacity}')
#         self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge -> positive, if discharge-> negative
#         self.current_capacity=updated_capacity# update capacity to current codition 
#         #print(f'Power charg/disch: {self.energy_change}')
#         #print('---****')

#     def _get_cost(self,energy):# calculate the cost depends on the energy change
#         cost = (energy**2)*self.degradation
#         return cost
    
#     @property
#     def SOC(self):
#         return self.current_capacity
    
#     def reset(self):
#         #self.current_capacity=round(np.random.uniform(0.2,0.8), 1)  # initial capcity rounded to 1 to get 0.2, 0.3,...
#         self.current_capacity=0.3      # set to 30%

class Battery():
    '''simulate a simple battery here'''
    def __init__(self,capacity, max_soc, min_soc, efficiency, degradation):
        self.capacity=capacity
        self.max_soc=max_soc
        # self.initial_capacity=parameters['initial_capacity']
        self.min_soc=min_soc # 0.2
        self.degradation=degradation # degradation cost 1.2
        self.efficiency=efficiency
    def step(self,action_battery):
        energy=action_battery
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge, positive, if discharge, negative
        self.current_capacity=updated_capacity# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        cost=energy**2*self.degradation
        return cost  
    @property
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity=np.random.uniform(0.2,0.8)
        # print(f'Battery initial soc: {self.current_capacity}')
        return self.current_capacity


class Grid:
    def __init__(self, tou_price):
        self.sell_prices = tou_price
        self.exchange_ability = 200    # 200 kW
        self.time = 0
    
    # to link self.timestep (which is current time_step) to self.time
    def set_time(self, time):
        self.time = time

    def _get_cost(self, p_ex):
        # print(f'price in grid class: {self.sell_prices[self.time]}, time_step: {self.time}')         
        return (self.sell_prices[self.time])*p_ex

# class DG:
#     def __init__(self, a_factor, b_factor):
#         self.a_factor=a_factor
#         self.b_factor=b_factor
#         self.last_step_output=None 

#     def step(self,dg_action):
#         self.current_output=dg_action

#     def _get_cost(self, output):
#         cost = ((self.a_factor*output**2)+(self.b_factor*output))
#         # print(cost)
#         return cost 
    
#     def reset(self):
#         self.current_output = 0
  

# class Generation:
#     def __init__(self, generation):
#         self.power = generation


#     def current_generation(self, time):
#         # We consider that we have 2 sources of power a constant source and a variable source
#         return self.power[time]


# class Load:
#     def __init__(self, base_load):
#         self.base_load = base_load

#     def current_load(self, time):
#         return self.base_load[time]

class MicroGridEnv(gym.Env):
    def __init__(self,**kwargs):

        # Get number of iterations and TCLs from the
        # parameters (we have to define it through kwargs because
        # of how Gym works...)
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.day0 = kwargs.get("day0", DEFAULT_DAY0)            # Day no. 0
        self.dayn = kwargs.get("dayn", DEFAULT_DAYN)    # Day no. depends on how much data model is being trained 
        self.tou_price = kwargs.get("tou_price", DEFAULT_TOU_PRICE)

        # Battery parameters
        self.bat_capacity = kwargs.get("battery_capacity", DEFAULT_BAT_CAPACITY)
        self.soc_max = kwargs.get("max_soc", DEFAULT_MAX_SOC)
        self.soc_min = kwargs.get("min_soc", DEFAULT_MIN_SOC)
        self.eff = kwargs.get("eff", DEFAULT_EFFICIENCY)
        self.deg = kwargs.get("deg", DEFAULT_DEGRADATION)
        self.soc_penalty = kwargs.get("soc_penalty", DEFAULT_PEN_SOC)

        # ── Projection reward hyper-params (new — thesis contribution) ────────
        self.proj_beta  = kwargs.get("proj_beta",  DEFAULT_PROJ_BETA)
        self.proj_gamma = kwargs.get("proj_gamma", DEFAULT_PROJ_GAMMA)
        # Track previous action so step() can compute the projection gap
        self._prev_action = 0.0
        self.last_gap = 0.0

        # reward_mode: 'projection' (thesis contribution, default) or
        # 'baseline' (paper reward, no gap penalty / alignment bonus)
        self.reward_mode = kwargs.get("reward_mode", "projection")
        assert self.reward_mode in ("projection", "baseline")
        # ─────────────────────────────────────────────────────────────────────

        # Penalty coefficient power imbalance
        # self.penalty_coefficient = kwargs.get("unb_penalty", DEFAULT_PEN_UNB)   #control soft penalty constraint
        
        # DG parameters
        # self.a = kwargs.get("a", DEFAULT_A)
        # self.b = kwargs.get("b", DEFAULT_B)

        # The current day: initally set to day no. 0
        self.day = self.day0

        # Setting first timestep of each time period
        self.time_step = 0

        # self.generation = Generation(kwargs.get("generation_data", DEFAULT_POWER_GENERATED))
        self.grid = Grid(tou_price=self.tou_price)
        self.battery = Battery(capacity=self.bat_capacity, max_soc=self.soc_max, min_soc=self.soc_min, \
                               efficiency=self.eff, degradation=self.deg)
        # self.dg = DG(a_factor=self.a, b_factor=self.b)

        # self.load = Load(kwargs.get('load_data', DEFAULT_BASE_LOAD))

        #self.action_space_sep = spaces.Box(low=0, high=1, dtype=np.float32,
                                       #shape=(13,))

        # Total 189 combination of actions = [(0, -80), (0, -60)...(10, -80),.. (20, 80)]
        #self.action_space = spaces.Discrete(55)
        self.action_space = spaces.Discrete(9)       # Total nine actions & are 1 dimensional see ACTIONS
        # self.action_space = spaces.Discrete(17)
        #self.action_space = spaces.Discrete(363)
        #self.action_space = spaces.Discrete(1353)

        # Observations: loads + battery soc + res generation + price + time-step of day (any array of shape = (5,))
        # self.observation_space = spaces.Box(low=-600, high=600, dtype=np.float32,
        #                                     shape=(5,))
        
        # Observations here are time_step of the day, SOC, Energy price for t and next t+23 (an array of shape=(26,))
        # self.observation_space = spaces.Box(low=0, high=1, dtype=np.float32,
        #                                     shape=(3,))
        
        # # Observations here are time_step of the day, SOC, Energy price for t and next t+23 (an array of shape=(26,))
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.float32,
                                            shape=(26,))



    def _projection_reward(self, sell_benefit, buy_cost, battery_cost, intended_action):
       
        # ── Baseline profit term (identical to paper) ─────────────────────────
        profit = float(sell_benefit) - float(buy_cost) - 0.01 * float(battery_cost)

        # ── Projection gap penalty ────────────────────────────────────────────
        # executed_action = energy that actually moved after SOC clamping
        executed_action = float(self.battery.energy_change)
        max_action      = float(max(abs(a) for a in ACTIONS))   # 80 kW

        gap      = abs(float(intended_action) - executed_action)
        gap_norm = gap / max_action          # normalise to [0, 1]

        self.last_gap = gap_norm

        gap_penalty = self.proj_beta * gap_norm

        # ── Price alignment bonus ─────────────────────────────────────────────
        # Charge when price is low, discharge when price is high.
        # Uses the same grid object as the baseline.
        price_t   = float(self.grid.sell_prices[self.grid.time])
        max_price = float(max(self.grid.sell_prices))
        p_norm    = price_t / max(max_price, 1e-6)   # normalised price ∈ [0, 1]

        if executed_action > 0:      # charging — good if price is low
            align = (1.0 - p_norm) * (executed_action / max_action)
        else:                        # discharging — good if price is high
            align = p_norm * (abs(executed_action) / max_action)

        align_bonus = self.proj_gamma * align

        # ── Final reward ──────────────────────────────────────────────────────
        reward = profit - gap_penalty + align_bonus

        # operation_cost mirrors baseline convention
        operation_cost = 0.01 * float(battery_cost) + float(buy_cost) - float(sell_benefit)

        return float(reward), float(operation_cost)

    def _build_state(self):
        """
        Return current state representation as one vector.
        Returns:
            state: 1D state vector, Loads, current battery soc, current power generation,
                 current price and current time (hour) of day
        """
        # current price + time of day (hour)
        # Scaling between 0 and 1
        # We need to standardize the generation and the price
        #print(f'Time_step in build state: {self.time_step}')
        # current_load = self.load.current_load(self.day*self.iterations + self.time_step) 
        # current_load = (current_load - min(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                 / (max(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations])\
        #                 - min(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations]))
        
        # current_load = np.float32(current_load/max(self.load.base_load))

        #print(f'max load: {max(self.load.base_load)}')
        # current_load = (current_load - np.average(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                 / np.std(self.load.base_load[self.day*self.iterations:self.day*self.iterations+self.iterations])

        #print(f'STEPS: {self.day*self.iterations+(self.time_step % 24)}')
        # current_generation = self.generation.current_generation(self.day*self.iterations + self.time_step)
        # current_generation = (current_generation-
        #                        np.average(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                       /np.std(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations])

        # current_generation = np.float32(current_generation/max(self.generation.power))

        # current_generation = (current_generation-
        #                        min(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations]))\
        #                       /(max(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations])\
        #                         - min(self.generation.power[self.day*self.iterations:self.day*self.iterations+self.iterations]))

        # Time steps we are not giving 'self.day*self.iterations+self.time_step' 
        # because right now we are only using same 24 hrs price data for whole training
        # change it when you will be trianing on years training data 
        
        # price_grid_sell = self.grid.sell_prices[self.time_step % 24]
        # price_grid_sell = (price_grid_sell -
        #          np.average(self.grid.sell_prices[0:self.iterations])) \
        #         / np.std(self.grid.sell_prices[0: self.iterations])

        # price_grid_sell = (price_grid_sell -
        #         min(self.grid.sell_prices[0:self.iterations])) \
        #         / (max(self.grid.sell_prices[0: self.iterations])\
        #            - min(self.grid.sell_prices[0: self.iterations]))

        # price_grid_sell = self.grid.sell_prices[self.day*self.iterations+self.time_step]
        # price_grid_sell = np.float32(price_grid_sell/max(self.grid.sell_prices))

        n_prices = len(self.grid.sell_prices)
        idx = self.day * self.iterations + self.time_step
        price_grid_sell = np.empty(24, dtype=np.float32)
        for k in range(24):
            price_grid_sell[k] = self.grid.sell_prices[min(idx + k, n_prices - 1)]
        price_grid_sell = np.float32(price_grid_sell / max(self.grid.sell_prices))
        #print(f'max price: {max(self.grid.sell_prices)}')
        
        time_step = np.float32((self.time_step)/(self.iterations-1))

        # Soc is also normalized to 1 by dividing max_soc=0.8
        soc = np.float32(self.battery.SOC/(self.battery.max_soc))

        # state = np.array([current_load, time_step, soc, current_generation,
        #                          price_grid_sell ])

        # state = np.array([time_step, soc, price_grid_sell])
        state = np.concatenate((np.array([time_step, soc]), price_grid_sell))
        #rint(f'States: {state}')
        return state

    def _build_info(self):
        """
        Return dictionary of misc. infos to be given per state.
        Here this means providing forecasts of future
        prices and temperatures (next 24h)
        """
        # temp_forecast = np.array(self.temperatures[self.time_step + 1:self.time_step + self.iterations+1])
        # return {"temperature_forecast": temp_forecast,
        #         "forecast_times": np.arange(0, self.iterations)}


    def step(self, action):
        """
        Arguments:
            action: A list.

        Returns:
            state: Current state
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (maximum number of iterations)
            info: None (not used here)
        """
        if type(action) is not list:
            action = ACTIONS[action]

        # print(f'Actions{self.time_step}: {action}')
        #print(f'Time_step in step: {self.time_step}')
        # Time steps we are not giving 'self.day*self.iterations+self.time_step' 
        # because right now we are only using same 24 hrs price data for whole training
        # change it when you will be trianing on years training data 
        # self.grid.set_time(self.time_step % 24)
        self.grid.set_time(self.day*self.iterations+self.time_step) # setting current time step for grid functions
        reward = 0

        # print(f'Day no. with time step in step function: {self.day*self.iterations+self.time_step}, day: {self.day}, timestep: {self.time_step}')

        # assigning action indeces to corresponding components
        # 0 -> DG and 1 -> Battery
        #print(f'Before Battery SOC: {self.battery.SOC}')
        # dg_action = action[0]
        # battery_action = action[1]
        battery_action = action
        #print(f'Battery action: {battery_action}')

        # ── Capture intended action BEFORE physics clamps it ──────────────────
        # This is the raw kW value the agent requested.  After Battery.step()
        # the actual energy_change may differ if SOC limits were hit.
        # _projection_reward() uses this to compute the projection gap.
        intended_action = float(battery_action)

        # DG action executed
        # self.dg.step(dg_action)

        # Battery action executed
        self.battery.step(battery_action)
        #print(f'Power charg/disch: {self.battery.energy_change}')
        #print(f'After Battery SOC: {self.battery.SOC}')

        # Get the energy generated by the DER without normalized form
        # der_generation = self.generation.current_generation(self.day*self.iterations + self.time_step)
        
        # Total current of all the DGs
        # total_current_output = der_generation + self.dg.current_output - self.battery.energy_change
        total_current_output = self.battery.energy_change     # Here the only generation or demand -> Ch/Dis
        # print(f'Total Gen: {total_current_output}')    # +ive/-ive means Ch & Dis 
        self.total_current_output = total_current_output
    
        # Total current load without normalized forms
        # total_loads = self.load.current_load(self.day*self.iterations + self.time_step)
        #total_loads = self.load.current_load(self.time_step%24)
        # print("Total loads",total_loads)

        # Computing unbalance
        # unbalance = total_current_output - total_loads
        # print(f'Unbalance: {unbalance}')

        # excess_penalty=0
        # deficient_penalty=0
        soc_penalty=0
        sell_benefit=0
        buy_cost=0
        # self.excess=0
        # self.shedding=0

        # logic here is: if unbalance >0 then it is production excess, so the excessed output should sold to power grid to get benefits 
        # if unbalance >= 0:# it is now in excess condition
        #     if unbalance <= self.grid.exchange_ability:
        #         sell_benefit = self.grid._get_cost(unbalance)   #sell money to grid is little [0.029,0.1]
        #     else:
        #         sell_benefit = self.grid._get_cost(self.grid.exchange_ability)
        #         #real unbalance that even grid could not meet 
        #         self.excess = unbalance - self.grid.exchange_ability
        #         #excess_penalty=self.excess*self.penalty_coefficient
        #         #excess_penalty=(self.excess**2)*self.penalty_coefficient
        #         excess_penalty = self.penalty_coefficient
        # else:# unbalance <0, its load shedding model, in this case, deficient penalty is used 
        #     if abs(unbalance)<=self.grid.exchange_ability:
        #         buy_cost=self.grid._get_cost(abs(unbalance))
        #     else:
        #         buy_cost=self.grid._get_cost(self.grid.exchange_ability)
        #         self.shedding=abs(unbalance)-self.grid.exchange_ability
        #         #deficient_penalty=self.shedding*self.penalty_coefficient
        #         #deficient_penalty=(self.shedding**2)*self.penalty_coefficient
        #         deficient_penalty = self.penalty_coefficient


        if total_current_output >= 0:# it is in charging mode (+ive energy change) so importing power from grid
            buy_cost = self.grid._get_cost(total_current_output)   #buy cost from grid
        else:# discharging mode <0 so selling power to grid, so it is selling  
            sell_benefit=self.grid._get_cost(abs(total_current_output))

        battery_cost=self.battery._get_cost(self.battery.energy_change) # we set it as 0 this time 
        # dg_cost = self.dg._get_cost(self.dg.current_output)
    
        # SOC constraints
        # if (self.battery.current_capacity < self.battery.min_soc or self.battery.current_capacity > self.battery.max_soc):
        #     soc_penalty = self.battery.pen_soc

        # if self.battery.current_capacity < self.battery.min_soc:
        #     soc_penalty = (self.battery.pen_soc)*(abs(self.battery.current_capacity-self.battery.min_soc))
        # else:
        #     soc_penalty = (self.battery.pen_soc)*(abs(self.battery.current_capacity-self.battery.max_soc))

        #print(f'soc penalties: {soc_penalty} and {self.battery.pen_soc}')

        #reward = -0.01*(battery_cost + dg_cost - sell_benefit + buy_cost) - (excess_penalty + deficient_penalty)
        # reward =  -1*(excess_penalty + deficient_penalty + soc_penalty)
        #reward = -0.01*(battery_cost + dg_cost - sell_benefit + buy_cost)
        #reward =  -1*(battery_cost + dg_cost - sell_benefit + buy_cost) - (excess_penalty + deficient_penalty + soc_penalty)

        # reward =  -0.01*(battery_cost - sell_benefit + buy_cost) - (soc_penalty)

        # reward =  -0.01*(battery_cost - sell_benefit + buy_cost)

        if self.reward_mode == "baseline":
            # ── BASELINE reward (paper) ───────────────────────────────────────
            reward = sell_benefit - buy_cost - 0.01*(battery_cost)
            self.operation_cost = 0.01 * float(battery_cost) + float(buy_cost) - float(sell_benefit)
            # gap is still tracked (for comparison plots) but not used in the reward
            executed_action = float(self.battery.energy_change)
            max_action = float(max(abs(a) for a in ACTIONS))
            self.last_gap = abs(float(intended_action) - executed_action) / max_action
        else:
            # ── PROJECTION reward (thesis contribution) ───────────────────────
            reward, proj_operation_cost = self._projection_reward(
                sell_benefit=sell_benefit,
                buy_cost=buy_cost,
                battery_cost=battery_cost,
                intended_action=intended_action,
            )
            self.operation_cost = proj_operation_cost

        self.op_co_woDC = buy_cost-sell_benefit

        
        '''
        - Unbalance within the main grid limits
        - Multiply by -1 to get +ive P_grid for import and -ive for export
        - Real unbalance that gets penalised beyond main grid limits
        '''
        # self.unbalance = (-1.0)*unbalance
        #print(f'Self Unbalance->grid: {self.unbalance}')         
        # self.real_unbalance = self.shedding + self.excess  

        #print(unbalance)

        # Proceed to next timestep.
        self.time_step += 1
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state()


        terminal = self.time_step == self.iterations
        # if terminal:

        #     # # reward if battery is charged
        #     # reward += abs(reward * self.battery.SOC / 2)
        info = self._build_info()
        return state, reward , terminal, info

    def reset(self,day=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if day == None:
            self.day= random.randint(self.day0,self.dayn-1)
        else:
            self.day = day
        # print("Day:", self.day)
        self.time_step = 0
        self.unbalance = 0
        self._prev_action = 0.0

        # Reset the battery to a random state
        self.in_SOC = self.battery.reset()
        
        # Reset DG to 0
        # self.dg.reset()

        return self._build_state()

    def reset_all(self,day=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if day == None:
            # self.day = random.randint(self.day0, self.dayn-1)
            self.day= self.day0
        else:
            self.day = day
        # print("Day:", self.day)
        self.time_step = 0
        in_SOC = self.battery.reset()
        self.high_price = 0
        self.loads.clear()
        self.loads = [self._create_load(*self._create_load_parameters()) for _ in range(self.num_loads)]

        return self._build_state()

    def render(self, mode = None):
        # LOADS_RENDER.append(self.load.current_load(self.day*self.iterations + self.time_step-1))
        BATTERY_RENDER.append(self.battery.SOC)
        BATTERY_CH_DCH_RANDER.append(self.battery.energy_change)
        # ENERGY_GENERATED_RENDER.append(self.generation.current_generation(self.day*self.iterations+self.time_step-1))
        # ENERGY_GRID_RENDER.append(self.unbalance)
        # ENERGY_R_UNBALANCE_RENDER.append(self.real_unbalance)
        # DG_GEN_RENDER.append(self.dg.current_output)

        # print(f'Grid price in render function: {self.grid.sell_prices[self.time_step-1]}, time_step: {self.time_step}')
        # GRID_PRICES_SELL_RENDER.append(self.grid.sell_prices[self.time_step-1])

        # print(f'Grid price in render function: {self.grid.sell_prices[self.day*self.iterations+self.time_step-1]}, time_step: {self.day*self.iterations+self.time_step-1}')
        
        GRID_PRICES_SELL_RENDER.append(self.grid.sell_prices[self.day*self.iterations+self.time_step-1])
        OPERATION_COST.append(self.operation_cost)
        OP_CO_woDC.append(self.op_co_woDC)

        if self.time_step==self.iterations:
            
            if mode == 'Eval':
                print(f'Initial SOC[0]: {self.in_SOC}')
                print(f'Bat Ch/Dch: {BATTERY_CH_DCH_RANDER}')
                print(f'Bat SOC: {BATTERY_RENDER}')
                # print(f'DG output: {DG_GEN_RENDER}')
                #fig=plt.figure()
                print(f'Total cost: {np.sum(np.array(OPERATION_COST))}')
                print(f'Total cost without degradation cost: {np.sum(np.array(OP_CO_woDC))}')


                #     # Define additional data
                #     additional_data = {
                #         'Parameter': ['Initial SOC', 'Total Cost', 'Total Cost without Degradation Cost'],
                #         'Value': [self.in_SOC, np.sum(np.array(OPERATION_COST)), np.sum(np.array(OP_CO_woDC))]  # Directly using the single value
                #     }

                #     # Convert BATTERY_CH_DCH_RANDER and BATTERY_RENDER to DataFrame
                #     df_ch_dch = pd.DataFrame({'Hour': np.arange(len(BATTERY_CH_DCH_RANDER)), 'Battery Ch/Dch': BATTERY_CH_DCH_RANDER})
                #     df_soc = pd.DataFrame({'Hour': np.arange(len(BATTERY_RENDER)), 'Battery SOC': BATTERY_RENDER})

                #     # Construct the filename
                #     filename = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\'
                #                 f'Res_LR{0.01}_'
                #                 f'SED{10000}_'
                #                 f'DF{0.99}_for_paper.xlsx')

                # # Check if file exists and create or append accordingly
                # try:
                #     with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                #         # Define sheet names
                #         sheet_name_params = 'Parameters'
                #         sheet_name_ch_dch = 'Battery_Ch_Dch'
                #         sheet_name_soc = 'Battery_SOC'
                #         # Write data to Excel
                #         pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name_params, index=False)
                #         df_ch_dch.to_excel(writer, sheet_name=sheet_name_ch_dch, index=False)
                #         df_soc.to_excel(writer, sheet_name=sheet_name_soc, index=False)
                #     print(f"Data successfully saved to {filename} in sheets: {sheet_name_params}, {sheet_name_ch_dch}, {sheet_name_soc}")
                # except Exception as e:
                #     print(f"An error occurred while saving data: {e}")



                # Define additional data
                # additional_data = {
                #     'Parameter': ['Initial SOC', 'Total Cost', 'Total Cost without Degradation Cost'],
                #     'Value': [self.in_SOC, np.sum(np.array(OPERATION_COST)), np.sum(np.array(OP_CO_woDC))]  # Directly using the single value
                # }

                # Convert BATTERY_CH_DCH_RANDER and BATTERY_RENDER to DataFrame
                # df_ch_dch = pd.DataFrame({'Hour': np.arange(len(BATTERY_CH_DCH_RANDER)), 'Battery Ch/Dch': BATTERY_CH_DCH_RANDER})
                # df_soc = pd.DataFrame({'Hour': np.arange(len(BATTERY_RENDER)), 'Battery SOC': BATTERY_RENDER})

                # Construct the filename
                # un comment this file name only

                # filename = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\'
                #             f'Res_LR{0.01}_'
                #             f'SED{10000}_'
                #             f'DF{1}.xlsx')

                # # Append additional data
                # with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                #     pd.DataFrame(additional_data).to_excel(writer, sheet_name='Parameters', index=False)
                #     df_ch_dch.to_excel(writer, sheet_name='Battery_Ch_Dch', index=False)
                #     df_soc.to_excel(writer, sheet_name='Battery_SOC', index=False)

                #     # Append additional data
                #     with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                #         pd.DataFrame(additional_data).to_excel(writer, sheet_name='Parameters', index=False)
                #         df_ch_dch.to_excel(writer, sheet_name='Battery_Ch_Dch', index=False)
                #         df_soc.to_excel(writer, sheet_name='Battery_SOC', index=False)


                '''
                Evaluation plots
                '''

                # # Create subplots
                # fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

                # # Set font properties globally
                # plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})

                # # Plot Energy Price
                # axs[0].grid(True, linestyle='-', color='lightgrey', zorder=1, which='both')
                # axs[0].bar(np.arange(0, 24), GRID_PRICES_SELL_RENDER, color='dodgerblue', zorder=3)  # Bars in the foreground
                # axs[0].set_title("Energy Price")
                # axs[0].set_ylabel("Energy Price €")

                # # Plot ESS SOC
                # axs[1].grid(True, linestyle='-', color='lightgrey', zorder=1, which='both')
                # axs[1].bar(np.arange(0, 24), BATTERY_RENDER, color='mediumseagreen', zorder=3)  # Bars in the foreground
                # axs[1].set_title("BSS SOC")
                # axs[1].set_ylabel("Energy (kWh)")

                # charge = np.maximum(BATTERY_CH_DCH_RANDER, 0)  # Only positive values for charging
                # discharge = np.minimum(BATTERY_CH_DCH_RANDER, 0)  # Only negative values for discharging

                # # Plot charge and discharge with different colors
                # axs[2].grid(True, linestyle='-', color='lightgrey', zorder=1, which='both')
                # axs[2].bar(np.arange(0, 24), charge, color='green', label='Charge', zorder=3)
                # axs[2].bar(np.arange(0, 24), discharge, color='red', label='Discharge', zorder=3)

                # # Title, labels, and legend
                # axs[2].set_title("BSS Charge/Discharge")
                # axs[2].set_ylabel("Power (kW)")
                # axs[2].legend(loc='upper right')
                # axs[2].legend(framealpha=0)

                # # Add x-label to the bottom subplot
                # axs[2].set_xlabel("Time (h)")

                # # Construct the filename
                # filename1 = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\Optuna_best_hyper-params.png')

                # Construct the filename
                # un comment this filename only 

                # filename1 = (f'C:\\Rl-Git-Projects\\BMS_Hyperpar_search\\results\\Loop1\\'
                #             f'FigSOC_LR{0.01}_'
                #             f'SED{10000}_'
                #             f'DF{1}.png')

                # # Adjust layout
                # plt.savefig(filename1, dpi=300, bbox_inches='tight')

                # # Adjust layout
                # plt.savefig(filename1, dpi=600, bbox_inches='tight')

                # plt.tight_layout()
                # plt.show()


            # LOADS_RENDER.clear()
            BATTERY_RENDER.clear()
            GRID_PRICES_SELL_RENDER.clear()
            # ENERGY_R_UNBALANCE_RENDER.clear()
            # ENERGY_GRID_RENDER.clear()
            # ENERGY_GENERATED_RENDER.clear()
            BATTERY_CH_DCH_RANDER.clear()
            # DG_GEN_RENDER.clear()
            OPERATION_COST.clear()
            OP_CO_woDC.clear()

        # return abs(self.real_unbalance), self.operation_cost
        return self.operation_cost
    
    '''
    Evaluation Plots for 1 complete day of scheduling
    # '''
    # def plot_ren(self, ENERGY_GENERATED_RENDER, ENERGY_GRID_RENDER, DG_GEN_RENDER, 
    #              BATTERY_CH_DCH_RANDER, ENERGY_R_UNBALANCE_RENDER, LOADS_RENDER):

    #     data = np.array([ENERGY_GENERATED_RENDER, ENERGY_GRID_RENDER, DG_GEN_RENDER, \
    #                      BATTERY_CH_DCH_RANDER, ENERGY_R_UNBALANCE_RENDER])

    #     data_shape = np.shape(data)

    #     # Take negative and positive data apart and cumulate
    #     def get_cumulated_array(data, **kwargs):
    #         cum = data.clip(**kwargs)
    #         cum = np.cumsum(cum, axis=0)
    #         d = np.zeros(np.shape(data))
    #         d[1:] = cum[:-1]
    #         return d  

    #     cumulated_data = get_cumulated_array(data, min=0)
    #     cumulated_data_neg = get_cumulated_array(data, max=0)

    #     # Re-merge negative and positive data.
    #     row_mask = (data<0)
    #     cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    #     data_stack = cumulated_data

    #     cols = ["g", "y", "b", "c", "m"]

    #     ax = plt.subplot(3,1, 3)
    #     ax.set_facecolor("silver")
    #     #ax.set_xlabel("Time (h)")
    #     ax.yaxis.grid(True)

    #     for i in np.arange(0, data_shape[0]):
    #         ax.bar(np.arange(data_shape[1]), data[i], bottom=data_stack[i], color=cols[i],)
        
    #     plt.plot(np.arange(data_shape[1]), np.array(LOADS_RENDER), color = 'r')
    #     plt.legend(['Load', 'RES', 'Imp/Exp', 'Dg_gen', 'Ch/Dch', 'P_unb'])
    #     plt.xlabel("Time (h)")
    #     plt.show()


    def close(self):
        """
        Nothing to be done here, but has to be defined
        """
        return

    def seedy(self, s):
        """
        Set the random seed for consistent experiments
        """
        random.seed(s)
        np.random.seed(s)


def _moving_average(values, window=12):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values

    window = max(1, min(int(window), values.size))
    if window == 1:
        return values

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode='valid')


if __name__ == '__main__':
    # ── Projection reward gap demo ─────────────────────────────────────────────
    print("=== Projection gap demo ===")
    demo = MicroGridEnv()
    demo.seedy(1)
    s = demo.reset(day=0)
    demo.battery.current_capacity = 0.21   # near lower SOC wall
    demo.grid.set_time(0)
    # manually step battery with -80 kW intended
    demo.battery.step(-80.0)
    r_proj, _ = demo._projection_reward(
        sell_benefit=demo.grid._get_cost(abs(demo.battery.energy_change)),
        buy_cost=0.0,
        battery_cost=demo.battery._get_cost(demo.battery.energy_change),
        intended_action=-80.0,
    )
    r_base = demo.grid._get_cost(abs(demo.battery.energy_change)) - 0.01 * demo.battery._get_cost(demo.battery.energy_change)
    print(f"  Intended action  : -80 kW (discharge)")
    print(f"  SOC before       : 0.21  (near lower wall 0.20)")
    print(f"  Executed (actual): {demo.battery.energy_change:.2f} kW  (clamped by physics)")
    print(f"  Gap              : {abs(-80.0 - demo.battery.energy_change):.2f} kW")
    print(f"  Baseline reward  : {r_base:+.4f}  (no gap signal)")
    print(f"  Projection reward: {r_proj:+.4f}  (gap penalised)")
    print()

    # Initialize the environment
    env = MicroGridEnv()
    env.seedy(1)
    # Save the rewards and costs in lists
    rewards = []
    costs = []
    for _ in range(5):
        # reset the environment to the initial state
        state = env.reset()
        # Call render to prepare the visualization

        # Interact with the environment (here we choose random actions) until the terminal state is reached

        while True:
            # Pick a random action (not by trained agent) from the action space (here we pick an index between 0 and 80) 
            action = env.action_space.sample()
            # action =[np.argmax(action[0:4]),np.argmax(action[4:9]),np.argmax(action[9:11]),np.argmax(action[11:])]
            #action=[1,2,0,0]
            # Using the index we get the actual action that we will send to the environment
            # print(ACTIONS[action])
            print(action)
            # Perform a step in the environment given the chosen action
            state, reward, terminal, _ = env.step(action)
            #state, reward, terminal, _ = env.step(list(action))

            rewards.append(reward)
            costs.append(env.operation_cost)
            env.render()
            if terminal:
                break
        print("Episode Reward:", sum(rewards[-env.iterations:]))
        print("Episode Cost:", sum(costs[-env.iterations:]))

    cumulative_rewards = np.cumsum(np.asarray(rewards, dtype=np.float32))
    cumulative_costs = np.cumsum(np.asarray(costs, dtype=np.float32))

    reward_ma = _moving_average(cumulative_rewards, window=12)
    cost_ma = _moving_average(cumulative_costs, window=12)
    reward_x = np.arange(len(cumulative_rewards))
    cost_x = np.arange(len(cumulative_costs))
    ma_x = np.arange(12 - 1, len(cumulative_rewards)) if len(cumulative_rewards) >= 12 else np.arange(len(reward_ma))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(reward_x, cumulative_rewards, label='Cumulative reward', color='tab:blue', linewidth=2)
    if reward_ma.size:
        axes[0].plot(ma_x, reward_ma, label='12-step moving average', color='tab:orange', linewidth=2)
    axes[0].set_title('Cumulative Reward')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(cost_x, cumulative_costs, label='Cumulative cost', color='tab:red', linewidth=2)
    if cost_ma.size:
        axes[1].plot(ma_x, cost_ma, label='12-step moving average', color='tab:green', linewidth=2)
    axes[1].set_title('Cumulative Cost')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Cost')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()