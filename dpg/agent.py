from environments import RunEnv2
import numpy as np
import random
from random_process import OrnsteinUhlenbeckProcess, RandomActivation
from time import time
import cPickle
from model import Agent, build_model


def elu(x):
    return np.where(x > 0, x, np.expm1(x))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class ActorNumpy(object):
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


def find_noise_delta(actor, states, target_d=0.2, tol=1e-3, max_steps=1000):
    orig_weights = actor.get_actor_weights(True)
    orig_act = actor.act_batch(states)

    sigma_min = 0.
    sigma_max = 100.
    sigma = sigma_max
    step = 0
    while step < max_steps:
        weights = [w + np.random.normal(scale=sigma, size=np.shape(w)).astype('float32')
                   for w in orig_weights]
        actor.set_actor_weights(weights, True)
        new_act = actor.act_batch(states)
        d = np.sqrt(np.mean(np.square(new_act - orig_act)))

        dd = d - target_d
        if np.abs(dd) < tol:
            actor.set_actor_weights(orig_weights, True)
            return sigma

        # too big sigma
        if dd > 0:
            sigma_max = sigma
        # too small sigma
        else:
            sigma_min = sigma
        sigma = sigma_min + (sigma_max - sigma_min) / 2
        step += 1

    actor.set_actor_weights(orig_weights, True)
    return sigma


def get_noisy_weights(params, sigma):
    weights = []
    for p in params:
        w = p.get_value()
        if p.name in ('gamma', 'beta'):
            weights.append(w)
        else:
            weights.append(w + np.random.normal(scale=sigma, size=np.shape(w)).astype('float32'))
    return weights


def run_agent(model_params, weights, state_transform, data_queue, weights_queue,
              process, global_step, updates, best_reward, max_steps=10000000):

    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(**model_params)
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.set_actor_weights(weights)

    env = RunEnv2(state_transform, max_obstacles=3, skip_frame=5)
    #random_process = RandomActivation(size=env.noutput)
    random_process = OrnsteinUhlenbeckProcess(theta=.1, mu=0., sigma=.2, size=env.noutput,
                                              sigma_min=0.15, n_steps_annealing=1e5)
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
        total_reward_original = 0.
        terminal = False
        steps = 0
        
        while not terminal:
            state = np.asarray(state, dtype='float32')

            action = actor.act(state)
            action += random_process.sample()

            next_state, reward, next_terminal, info = env.step(action)
            total_reward += reward
            total_reward_original += info['original_reward']
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

        states_np = np.asarray(states).astype('float32')
        data = (states_np,
                np.asarray(actions).astype('float32'),
                np.asarray(rewards).astype('float32'),
                np.asarray(terminals).astype('float32'),
                )
        # send data for training
        data_queue.put((process, data))

        # receive weights and set params to weights
        weights = weights_queue.get()
        actor.set_actor_weights(weights)
        sigma = find_noise_delta(actor, states_np, random_process.current_sigma)
        weights = get_noisy_weights(actor.params_actor, sigma)
        actor.set_actor_weights(weights)

        # clear buffers
        del states[:]
        del actions[:]
        del rewards[:]
        del terminals[:]

        report_str = 'Global step: {}, steps/sec: {:.2f}, updates: {}, episode len {}, ' \
                     'reward: {:.2f}, original_reward {:.4f}; best reward: {:.2f}'. \
            format(global_step.value, 1. * global_step.value / (time() - start), updates.value, steps,
                   total_reward, total_reward_original, best_reward.value)
        print report_str

        with open('report.log', 'a') as f:
            f.write(report_str + '\n')
