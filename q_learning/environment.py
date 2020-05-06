import gym
from gym import spaces
from gym.utils import seeding
import itertools
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

# Discharges the batteries. We consume less energy that we would
# normally do.
def discharge(env, load, timestamp=None):
    discharge_amount = min(load, env.battery_charge_speed, env.battery_level)
    env.battery_level -= discharge_amount
    env.history.append(load - discharge_amount)

# Charges the batteries. Meaning, the system consumption increases
# by the battery charging speed.
def charge(env, load, timestamp=None):
    charge_amount = min(env.battery_charge_speed, env.battery_capacity - env.battery_level)
    env.battery_level += charge_amount
    env.history.append(load + charge_amount)

# Does nothing. The real consumption is the same as original one.
def wait(env, load, timestamp=None):
    env.history.append(load)

ACTIONS = [wait, charge, discharge]
ACTION_NAMES = ['wait', 'charge', 'discharge']

def cost(load):
    return np.sum(5 + 0.5 * load + 0.05 * load ** 2)

def cost_normalized(env, consumption_without_bot, consumption_with_bot):
    return cost_diff(env, consumption_without_bot, consumption_with_bot) / 100

def cost_diff(env, consumption_without_bot, consumption_with_bot):
    return cost(consumption_without_bot) - cost(consumption_with_bot)

def cost_by_24h_diff(env, consumption_without_bot, consumption_with_bot):
    if env.cur_step % 24 != 0:
        return 0
    cost_sum = 0
    for i in range(env.cur_step - 23, env.cur_step, 1):
        cost_sum += (cost(env.consumption_history[i]) - cost(env.history[i]))
    return cost_sum

class Env(gym.Env):
    def __init__(self, train_data, battery_level=200, reward_func=cost_normalized):
        self.battery_level = battery_level
        self.consumption_history = train_data
        self.battery_capacity = 400
        self.battery_charge_speed = 100
        self.cur_step = None
        self.n_step = len(train_data)
        self.reward_func = reward_func
        self.all_history = []

        self.action_space = spaces.Discrete(3)

        max_consumption = self.consumption_history.max()
        # min_consumption = self.consumption_history.min(axis=1) / 1.2
        battery_range = [0, self.battery_capacity]
        consumption_range = [0, max_consumption * 1.5]
        self.observation_space = spaces.MultiDiscrete([battery_range, consumption_range])

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = 0
        self.history = [self.consumption_history[0]]
        # self.battery_level = 0
        return self._get_current_state()

    def _get_current_state(self):
        return [self.battery_level, self._consumption_to_pay()]

    def _user_consumption(self):
        return self.consumption_history[self.cur_step]

    def _consumption_to_pay(self):
        return self.history[self.cur_step]

    def _real_action(self, action):
        # We are unable to charge when the battery is full, so we wait
        # The same regarding discharge
        if self.battery_level == 0 and action == 2:
            return 0
        if self.battery_level == self.battery_capacity and action == 1:
            return 0
        return action

    def _step(self, action):
        assert self.action_space.contains(action)
        _prev_battery_level = self.battery_level

        self.cur_step += 1
        real_action = self._real_action(action)
        ACTIONS[real_action](self, self._user_consumption())
        reward = self.reward_func(self, self._user_consumption(), self._consumption_to_pay())
        info = {'cur_val': cost_diff(self, self._user_consumption(), self._consumption_to_pay()) }

        # self.all_history.append(
        #     (
        #         self.cur_step,
        #         _prev_battery_level,
        #         self.consumption_history[self.cur_step - 1],
        #         ACTION_NAMES[action],
        #         ACTION_NAMES[real_action],
        #         self.battery_level,
        #         self._user_consumption(),
        #         self._consumption_to_pay(),
        #         cost_diff(self, self._user_consumption(), self._consumption_to_pay()),
        #         reward
        #     )
        # )
        done = self.cur_step == self.n_step - 1
        return self._get_current_state(), reward, done, info
