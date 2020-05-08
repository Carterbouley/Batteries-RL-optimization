from collections import deque
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(n_obs, n_action, display=True, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
    """ A multi-layer perceptron """
    # import pdb;pdb.set_trace();
    model = Sequential()
    model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))
    model.add(Dense(n_action, activation='linear'))
    model.compile(loss=loss, optimizer=Adam())
    if display: print(model.summary())
    return model

class MlpDqn(nn.Module):
    def __init__(self, n_state_input, n_hidden_1, n_hidden_2, n_actions):
        super(MlpDqn, self).__init__()
        self.fc1 = nn.Linear(n_state_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.out = nn.Linear(n_hidden_2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def mlp2(n_obs, n_action, display=True, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
    return MlpDqn(n_obs, n_neuron_per_layer, n_neuron_per_layer, n_action)
class DQNAgent(object):
    """ A simple Deep Q agent """
    def __init__(self, state_size, action_size, display=True, memory_size=2000, mlp=mlp2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size, display)
        self.tmodel = mlp(state_size, action_size, display=False)

    # Applies the action chozen at the previous timestamp to the
    # current time and load data. For example, if 1 hour ago we choose to
    # charge the battery, now we need to increase the default concumption
    # by that value. Then, write it to the alternative history.
    # Afteer saving history, it also chooses action for the next 1 hour.
    def tick(self, timestamp, load):
        self.action(self, timestamp, load)
        self.action = self.policy(self, timestamp, load)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.Tensor(state)
        optimal_action_index = self.model(state).max(0).indice
        return self.actions[optimal_action_index]

    def copy_to_target(self):
        self.tmodel.load_state_dict(self.model.state_dict())
        # # self.tmodel.set_weights(self.model.get_weights())
        # weights = self.model.get_weights()
        # target_weights = self.tmodel.get_weights()
        # for i in range(len(target_weights)):
        #     target_weights[i] = weights[i]
        # self.tmodel.set_weights(target_weights)

    def replay(self, batch_size=32, copy_weights=False):
        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = random.sample(self.memory, batch_size)

        states = torch.Tensor([tup[0][0] for tup in minibatch])
        actions = torch.LongTensor([tup[1] for tup in minibatch])
        rewards = torch.Tensor([tup[2] for tup in minibatch])
        next_states = torch.Tensor([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # # Q(s', a)
        # target = rewards + self.gamma * np.amax(self.tmodel.predict(next_states), axis=1)
        # # end state target is reward itself (no lookahead)
        # target[done] = rewards[done]

        # # Q(s, a)
        # target_f = self.tmodel.predict(states)
        # # tt= target_f.copy()
        # # make the agent to approximately map the current state to future discounted reward
        # target_f[range(batch_size), actions] = target

        # history = self.model.fit(states, target_f, epochs=1, verbose=0)

        all_qvs = self.model(states)
        # Shape: torch.Size([32, 3])
        # import pdb; pdb.set_trace();
        state_action_values = all_qvs.gather(1, actions)
        # RuntimeError: invalid argument 4: Index tensor must have same dimensions as input tensor at ../aten/src/TH/generic/THTensorEvenMoreMath.cpp:638

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = self.tmodel(next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = rewards + (next_state_values * self.gamma)

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        if copy_weights:
            self.copy_to_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
