import pickle
import re
import time
import numpy as np

from .agent import DQNAgent
from .environment import Env, cost_diff, cost_by_24h_diff
from .utils import get_scaler, make_dir

def test(df, weights, scaler, initial_battery_level=200):
    env = Env(df['load'].values, initial_battery_level)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, display=False)
    agent.model.set_weights(weights)
    agent.epsilon = 0
    state = env.reset()
    state = scaler.transform([state])
    sum_reward = 0

    for i in range(env.n_step):
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = scaler.transform([next_state])
        sum_reward += info['cur_val']
        if i % 1000 == 0:
            print("battery: ", action, next_state)
        if done:
            print("Test Economy: {:,.2f}".format(sum_reward))
            print()
            break

def run(df, weights_dir, portfolio_dir, train_coef=0.8, initial_battery_level=200,
        mode='train', episode=5, batch_size=32, memory_size=2000):
    train_size = int(df.shape[0] * train_coef)
    print("Train data size", train_size)
    train_data = df[:train_size]
    test_data = df[train_size:]
    env = Env(train_data['load'].values,
              initial_battery_level,
              reward_func=cost_by_24h_diff)
    state_size = env.observation_space.shape
    print("State size:", state_size)
    action_size = env.action_space.n
    print("Action size:", action_size)
    agent = DQNAgent(state_size, action_size, memory_size=memory_size)
    scaler = get_scaler(env)
    portfolio_value = []

    make_dir(weights_dir)
    make_dir(portfolio_dir)
    weights_file = None
    # weights_file = f'{weights_dir}/202005062112-dqn.h5'

    if weights_file:
        timestamp = re.findall(r'\d{12}', weights_file)[0]
    else:
        timestamp = time.strftime('%Y%m%d%H%M')

    if mode == 'test':
        # import pdb; pdb.set_trace()
        agent.epsilon = 0.01
        agent.load(weights_file)
        env = Env(test_data['load'].values, initial_battery_level)

    for e in range(episode):
        state = env.reset()
        state = scaler.transform([state])
        sum_reward = 0
        losses = []
        for i in range(env.n_step):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            sum_reward += info['cur_val']
            # print(next_state, reward, done, info)
            next_state = scaler.transform([next_state])
            if mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Epoch: {}/{}, Train economy : {:,.2f}".format(
                e + 1, episode, sum_reward))
                portfolio_value.append(info['cur_val'])
                test(test_data, agent.tmodel.get_weights(), scaler)
                break
            if mode == 'train' and len(agent.memory) > batch_size:
                history = agent.replay(batch_size, i % 100 == 0)
                losses.append(history.history['loss'][0])
                if i % 2000 == 0:
                    print("Loss: {:,.2f}".format(np.array(losses).mean()))
                    losses = []
        if mode == 'train' and (e + 1) % 2 == 0:
            agent.save('{}/{}-dqn.h5'.format(weights_dir, timestamp))
        with open('{}/{}-{}.p'.format(portfolio_dir, timestamp, mode), 'wb') as fp:
            pickle.dump(portfolio_value, fp)
