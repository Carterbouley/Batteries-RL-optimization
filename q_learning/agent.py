from collections import deque
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
    """ A multi-layer perceptron """
    model = Sequential()
    model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))
    model.add(Dense(n_action, activation='linear'))
    model.compile(loss=loss, optimizer=Adam())
    print(model.summary())
    return model

class DQNAgent(object):
    """ A simple Deep Q agent """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)
        self.tmodel = mlp(state_size, action_size)

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
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32, copy_weights=False):
        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        target = rewards + self.gamma * np.amax(self.tmodel.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.tmodel.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target

        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        # import pdb; pdb.set_trace();

        if copy_weights:
            self.tmodel.set_weights(self.model.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
