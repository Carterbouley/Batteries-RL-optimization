import pickle
import re
import time

from .agent import DQNAgent
from .environment import Env, cost_diff
from .utils import get_scaler, make_dir

def run(df, weights_dir, portfolio_dir, train_coef=0.8, initial_battery_level=200,
        mode='train', episode=2, batch_size=32):
    train_size = int(df.shape[0] * train_coef)
    print("Train data size", train_size)
    train_data = df[:train_size]
    test_data = df[train_size:]
    env = Env(train_data['load'].values, initial_battery_level)
    state_size = env.observation_space.shape
    print("State size:", state_size)
    action_size = env.action_space.n
    print("Action size:", action_size)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    portfolio_value = []

    make_dir(weights_dir)
    make_dir(portfolio_dir)

    timestamp = time.strftime('%Y%m%d%H%M')

    if mode == 'test':
        # import pdb; pdb.set_trace()
        weights = './weights/202005031527-dqn.h5'
        agent.epsilon = 0.01
        agent.load(weights)
        timestamp = re.findall(r'\d{12}', weights)[0]
        env = Env(test_data['load'].values, initial_battery_level)

    for e in range(episode):
        state = env.reset()
        state = scaler.transform([state])
        sum_reward = 0
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
                print("episode: {}/{}, episode end value: {:,.2f}".format(
                e + 1, episode, sum_reward))
                portfolio_value.append(info['cur_val'])
                break
            if mode == 'train' and len(agent.memory) > batch_size:
                history = agent.replay(batch_size, i % 100 == 0)
                if i % 1000 == 0:
                    print("{:,.2f}".format(history.history['loss'][0]))
        if mode == 'train' and (e + 1) % 10 == 0:
            agent.save('{}/{}-dqn.h5'.format(weights_dir, timestamp))
        with open('{}/{}-{}.p'.format(portfolio_dir, timestamp, mode), 'wb') as fp:
            pickle.dump(portfolio_value, fp)
