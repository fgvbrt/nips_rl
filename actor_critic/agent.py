import numpy as np
import random
from time import time
import cPickle
from scipy import signal
from osim.env import RunEnv


class RunEnv2(RunEnv):
    def _step(self, action):
        s, r, t, info =super(RunEnv2, self)._step(action)
        return s, 100*r, t, info


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype('float32')


def gaussian(mu, sigma, sample=True):
    if sample:
        return np.random.normal(mu, sigma).astype('float32')
    else:
        return mu.astype('float32')


def elu(x):
    return np.where(x > 0, x, np.expm1(x))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softplus(x):
    return np.log(1. + np.exp(x))


class Actor(object):
    def __init__(self, weights, activation):
        self.set_weights(weights)
        self.activation = activation

    def set_weights(self, weights):
        self.weights = weights[:-2]
        self.mean = weights[-2]
        self.var = weights[-1]
        self.num_layers = len(self.weights)/2

    def save_weights(self, fname):
        with open(fname, 'wb') as f:
            cPickle.dump(self.weights + [self.mean, self.var], f, -1)

    def choose_action(self, s, sample=False):
        x = (s - self.mean) / (np.sqrt(self.var) + 1e-5)
        for i in xrange(self.num_layers):
            x = np.dot(x, self.weights[2*i]) + self.weights[2*i+1]
            if i != self.num_layers - 1:
                x = self.activation(x)

        mu = x[:len(x)/2]
        #mu = sigmoid(mu)
        std = softplus(x[len(x)/2:])
        a = gaussian(mu, std, sample)
        return np.clip(a, 0, 1)


def run_agent(actor, data_queue, weights_queue, gamma,
              batch_episodes, global_step, best_test_reward,
              max_steps=10000000):

    env = RunEnv2(visualize=False)

    # prepare buffers for data
    states = []
    actions = []
    rewards = []
    S = []
    A = []
    T = []

    total_episodes = 0
    start = time()
    while global_step.value < max_steps:
        for i in xrange(batch_episodes):
            seed = random.randrange(2 ** 32 - 2)
            state = env.reset(seed=seed)

            total_reward = 0.
            terminal = False
            steps = 0
            while not terminal:
                state = np.asarray(state, dtype='float32')

                action = actor.choose_action(state, True)

                next_state, reward, next_terminal, _ = env.step(action)
                total_reward += reward
                steps += 1
                global_step.value += 1

                # add data to buffers
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                terminal = next_terminal

                if terminal:
                    break

            total_episodes += 1

            # report progress
            report_str = 'Global step: {}, steps/sec: {:.2f}, episode length {}, reward: {:.2f}, best reward: {:.2f}'. \
                format(global_step.value, 1. * global_step.value / (time() - start), steps, total_reward,
                       best_test_reward.value)
            print report_str

            with open('report.log', 'a') as f:
                f.write(report_str + '\n')

            # add data to batch buffers
            S.append(np.asarray(states).astype('float32'))
            A.append(np.asarray(actions).astype('float32'))
            T.append(np.asarray(discount(rewards, gamma)).astype('float32'))

            # clear episode buffers
            del states[:]
            del actions[:]
            del rewards[:]

        # send data for training
        data = (np.concatenate(S), np.concatenate(A), np.concatenate(T))
        data_queue.put(data)

        # clear batch buffers
        del S[:]
        del A[:]
        del T[:]

        # receive weights and set params to weights
        weights = weights_queue.get()
        actor.set_weights(weights)
