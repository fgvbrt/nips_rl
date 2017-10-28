import numpy as np
from model import build_model
from time import time
import scipy.signal
from environments import RunEnv2


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


def run(model_params, weights_shared, weights_shared_target,
        state_transform, global_step, best_reward,
        n_steps=20, max_steps=50000, gamma=0.99):

    # init environment
    env = RunEnv2(state_transform, max_obstacles=3, skip_frame=1)

    # init model
    steps_fn, policy_fn, val_fn, target_update_fn, \
        params, params_target = build_model(**model_params)

    # set initial weights
    set_theano_weights(weights_shared, params)
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

            for _ in range(n_steps):
                # do action
                state = np.asarray(state, dtype='float32')
                action = policy_fn([state])[0]

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
            update_shared_weights(weights_shared, steps)
            set_theano_weights(weights_shared, params)

            # update target network
            target_update_fn()
            set_shared_weights(weights_shared_target, params_target)

            global_step.value += len(rewards) - 1
            mean_val += val_fn([states[len(states) // 2]])[0]

            # clear buffers
            del states[:]
            del rewards[:]
            del actions[:]

            if terminal:
                epoch += 1
                break

        current_episode += 1
        report_str = 'Global step: {}, steps/sec: {:.2f}, mean value: {:.2f},  episode length {}, reward: {:.2f}'.\
            format(global_step.value, 1.*global_step.value/(time() - start), mean_val/step, step, total_reward)
        print(report_str)

        with open('report.log', 'a') as f:
            f.write(report_str + '\n')
