import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

from environments import RunEnv2
import numpy as np
import random
from random_process import OrnsteinUhlenbeckProcess, RandomActivation
from time import time
from model import Agent, build_model
import Pyro4
import argparse
from state import StateVelCentr


def set_params_noise(actor, states, target_d=0.2, tol=1e-3, max_steps=1000):
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
            break

        # too big sigma
        if dd > 0:
            sigma_max = sigma
        # too small sigma
        else:
            sigma_min = sigma
        sigma = sigma_min + (sigma_max - sigma_min) / 2
        step += 1


@Pyro4.expose
class Sampler(object):

    def __init__(self):
        self.actor = None
        self.rand_process = None
        self.env = None
        self.env_params = None
        self.total_episodes = 0
        self.action_noise = True
        self.best_reward = None
        self.last_states = None

    def set_best_reward(self, val):
        self.best_reward = val

    def create_actor(self, params):
        train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
            build_model(**params)
        self.actor = Agent(actor_fn, params_actor, params_crit)

    def set_actor_weights(self, weights):
        self.actor.set_actor_weights(weights)

    def create_rand_process(self, params):
        self.rand_process = OrnsteinUhlenbeckProcess(**params)

    def create_env(self, params):
        state_transform = StateVelCentr(**params['state_transform'])
        self.env_params = params['env']
        self.env_params['state_transform'] = state_transform
        self.env = RunEnv2(**self.env_params)

    def initialize(self, config, weights):
        self.create_actor(config['model_params'])
        self.create_env(config['env_params'])
        self.create_rand_process(config['rand_process'])
        self.set_actor_weights(weights)

    @property
    def initialized(self):
        return self.actor is not None and self.rand_process is not None and self.env is not None

    def _sample_params_noise(self):
        action_noise = np.random.rand() < 0.7
        if self.last_states is not None and self.initialized and not action_noise:
            set_params_noise(self.actor, self.last_states, self.rand_process.current_sigma)
        return action_noise

    def run_episode(self):

        # prepare buffers for data
        states = []
        actions = []
        rewards = []
        terminals = []

        start = time()
        action_noise = self._sample_params_noise()

        seed = random.randrange(2 ** 32 - 2)
        state = self.env.reset(seed=seed, difficulty=2)
        self.rand_process.reset_states()

        total_reward = 0.
        total_reward_original = 0.
        terminal = False
        steps = 0

        while not terminal:
            state = np.asarray(state, dtype='float32')
            action = self.actor.act(state)

            if action_noise:
                action += self.rand_process.sample()

            next_state, reward, next_terminal, info = self.env.step(action)
            total_reward += reward
            total_reward_original += info['original_reward']
            steps += 1

            # add data to buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            terminal = next_terminal

            if terminal:
                break

        self.total_episodes += 1

        # add data to buffers after episode end
        states.append(state)
        actions.append(np.zeros(self.env.noutput))
        rewards.append(0)
        terminals.append(terminal)

        ret = {
            'states': np.asarray(states).astype(np.float32).tolist(),
            'actions': np.asarray(actions).astype(np.float32).tolist(),
            'rewards': np.asarray(rewards).astype(np.float32).tolist(),
            'terminals': np.asarray(terminals).tolist(),
            'total_reward': total_reward,
            'total_reward_original': total_reward_original,
            'steps': steps,
            'time_took': time() - start,
            'action_noise': action_noise
        }

        # if reward is higher than best give it to coordinator to check
        if self.best_reward is not None and total_reward > self.best_reward and total_reward > 0:
            ret['weights'] = [w.tolist() for w in self.actor.get_actor_weights()]

        if self.total_episodes % 100 == 0:
            self.env = RunEnv2(**self.env_params)

        # need to keep it for selecting param noise
        self.last_states = ret['states']

        return ret


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--name', type=str, default='1', help="Name of server")
    parser.add_argument('--host', type=str, default='localhost', help="Host name.")
    parser.add_argument('--port', type=int, default=9091, help="Host port.")
    parser.add_argument('--nathost', type=str, default=None, help="Nat name.")
    parser.add_argument('--natport', type=int, default=None, help="Nat port.")
    parser.add_argument('--ns_host', type=str, default='94.45.222.176', help="Name server host.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sampler = Sampler()

    # for example purposes we will access the daemon and name server ourselves and not use serveSimple
    with Pyro4.Daemon(host=args.host, port=args.port, nathost=args.nathost, natport=args.natport) as daemon:
        uri = daemon.register(sampler)
        name = "sampler.{}.port.{}".format(args.name, args.port)
        with Pyro4.locateNS(host=args.ns_host) as ns:
            ns.register(name, uri)
        print("Sampler ready: name {} uri {}".format(name, uri))
        daemon.requestLoop()
