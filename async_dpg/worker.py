import numpy as np
from model import build_model
from time import time
import scipy.signal
import cPickle
from osim.env import RunEnv
from random_process import OrnsteinUhlenbeckProcess

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype('float32')


def set_theano_weights(weights_shared, params):
    for w, p in zip(weights_shared, params):
        p.set_value(w)


def update_shared_weights(weights_shared, steps):
    for w, s in zip(weights_shared, steps):
        w -= s


def set_shared_weights(weights_shared, params):
    for w, p in zip(weights_shared, params):
        w[:] = p.get_value()


class RunEnv2(RunEnv):
    def _step(self, action):
        s, r, t, info =super(RunEnv2, self)._step(action)
        return s, 100*r, t, info


def run(process, weights_shared_cur, weights_shared_target,
        global_step, best_reward,
        weights_save_intervsal, num_test_episodes,
        n_steps=20, max_steps=50000, gamma=0.99):

    # init environment
    env = RunEnv2(visualize=False)

    # exploration
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=18,
                                              sigma_min=1e-6, n_steps_annealing=2e6)
    # init model
    steps_fn, policy_fn, val_fn, target_update_fn, \
        params, params_target = build_model(41, 18)

    # set initial weights
    set_theano_weights(weights_shared_cur, params)
    set_theano_weights(weights_shared_target, params_target)

    # state and rewards for training
    states = []
    actions = []
    rewards = []

    epoch = 0
    current_episode = 0
    start = time()
    while global_step.value < max_steps:
        state = env.reset()

        total_reward = 0.
        terminal = False
        step = 0
        mean_val = 0.

        while not terminal:

            for _ in xrange(n_steps):
                # do action
                state = np.asarray(state, dtype='float32')
                action = policy_fn([state])[0]
                action += random_process.sample()

                states.append(state)
                actions.append(action)

                state, reward, terminal, _ = env.step(action)
                state = np.asarray(state, dtype='float32')

                total_reward += reward
                step += 1

                rewards.append(reward)

                if terminal:
                    break

            if terminal:
                rewards.append(0)
            else:
                rewards.append(val_fn([state])[0])

            v_batch = discount(rewards, gamma)[:-1]
            # training step
            steps = steps_fn(states, actions, v_batch)

            # update current network
            update_shared_weights(weights_shared_cur, steps)
            set_theano_weights(weights_shared_cur, params)

            # update target network
            target_update_fn()
            set_shared_weights(weights_shared_target, params_target)

            global_step.value += len(rewards) - 1
            mean_val += val_fn([states[len(states)/2]])[0]

            # clear buffers
            del states[:]
            del rewards[:]
            del actions[:]

            if terminal:
                epoch += 1
                break

        current_episode += 1
        if process == 0 and current_episode % weights_save_intervsal == 0:
            total_reward = 0
            for ep in range(num_test_episodes):
                state = env.reset()
                while True:
                    state = np.asarray(state, dtype='float32')
                    action = policy_fn([state])[0]
                    state, reward, terminal, _ = env.step(action)
                    total_reward += reward
                    if terminal:
                        break

            mean_reward = 1.*total_reward/num_test_episodes
            print 'test reward mean', mean_reward
            if mean_reward > best_reward.value:
                best_reward.value = mean_reward
                param_values = [p.get_value() for p in params]
                with open('weights_steps_{}_reward_{}.pkl'.format(global_step.value, int(mean_reward)), 'wb') as f:
                    cPickle.dump(param_values, f, -1)

        report_str = 'Global step: {}, steps/sec: {:.2f}, mean value: {:.2f},  episode length {}, reward: {:.2f}, best reward: {:.2f}'.\
            format(global_step.value, 1.*global_step.value/(time() - start), mean_val/step, step, total_reward, best_reward.value)
        print report_str

        if process == 0:
            with open('report.log', 'a') as f:
                f.write(report_str + '\n')
