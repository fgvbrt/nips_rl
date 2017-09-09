from environments import RunEnv2
import numpy as np
import random
from random_process import OrnsteinUhlenbeckProcess
from time import time
import cPickle


def elu(x):
    return np.where(x > 0, x, np.expm1(x))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Actor(object):
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation

    def set_weights(self, new_weights):
        self.weights = new_weights

    def save_weights(self, fname):
        with open(fname, 'wb') as f:
            cPickle.dump(self.weights, f, -1)

    def act(self, s):
        x = s
        num_layers = len(self.weights)/ 2
        for i in xrange(num_layers):
            x = np.dot(x, self.weights[2*i]) + self.weights[2*i+1]
            if i != num_layers - 1:
                x = self.activation(x)

        return sigmoid(x)


def run_agent(actor, state_transform, data_queue, weights_queue, process, global_step, updates, best_test_reward,
              testing_period, num_test_episodes, max_steps=10000000):

    env = RunEnv2(state_transform)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput,
                                              sigma_min=0.05, n_steps_annealing=1e6)

    # prepare buffers for data
    states = []
    actions = []
    rewards = []
    terminals = []

    total_episodes = 0
    start = time()
    while global_step.value < max_steps:
        seed = random.randrange(2**32-2)
        state = env.reset(seed=seed, difficulty=2)
        random_process.reset_states()

        total_reward = 0.
        terminal = False
        steps = 0
        while not terminal:
            state = np.asarray(state, dtype='float32')

            action = actor.act(state)
            action += random_process.sample()

            next_state, reward, next_terminal, _ = env.step(action)
            total_reward += reward
            steps += 1
            global_step.value += 1

            # add data to buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            terminal = next_terminal

            if terminal:
                break

        total_episodes += 1

        # add data to buffers after episode end
        states.append(state)
        actions.append(np.zeros(env.noutput))
        rewards.append(0)
        terminals.append(terminal)
        data = (np .asarray(states).astype('float32'),
                np.asarray(actions).astype('float32'),
                np.asarray(rewards).astype('float32'),
                np.asarray(terminals).astype('float32'),
                )
        # send data for training
        data_queue.put((process, data))

        # clear buffers
        del states[:]
        del actions[:]
        del rewards[:]
        del terminals[:]

        # receive weights and set params to weights
        weights = weights_queue.get()
        actor.set_weights(weights)

        if process == 0 and total_episodes % testing_period == 0:
            total_test_reward = 0
            for ep in range(num_test_episodes):
                state = env.reset(difficulty=2)

                while True:
                    state = np.asarray(state, dtype='float32')
                    action = actor.act(state)
                    state, reward, terminal, _ = env.step(action)
                    total_test_reward += reward
                    if terminal:
                        break

            mean_reward = 1. * total_test_reward / num_test_episodes
            print 'test reward mean', mean_reward
            if mean_reward > best_test_reward.value:
                best_test_reward.value = mean_reward
                fname = 'weights/weights_steps_{}_reward_{}.pkl'.format(global_step.value, int(mean_reward))
                actor.save_weights(fname)

        report_str = 'Global step: {}, steps/sec: {:.2f}, updates: {}, episode len {}, reward: {:.2f}, best reward: {:.2f}'. \
            format(global_step.value, 1. * global_step.value / (time() - start), updates.value, steps, total_reward, best_test_reward.value)
        print report_str

        with open('report.log', 'a') as f:
            f.write(report_str + '\n')
