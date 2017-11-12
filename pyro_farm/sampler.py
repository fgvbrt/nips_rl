import os
os.environ['OMP_NUM_THREADS'] = '1'

from environments import RunEnv2
import numpy as np
import random
from time import time
from model import Agent, build_model
import Pyro4
import argparse
from state import StateVelCentr


@Pyro4.expose
class Sampler(object):

    def __init__(self):
        self.actor = None
        self.env = None
        self.env_params = None
        self.total_episodes = 0

    def create_actor(self, params):
        _, _, _, actor_fn, _, params_actor, params_crit, _, _ = \
            build_model(**params)
        self.actor = Agent(actor_fn, params_actor, params_crit)

    def set_actor_weights(self, weights):
        self.actor.set_actor_weights(weights)

    def create_env(self, params):
        state_transform = StateVelCentr(**params['state_transform'])
        self.env_params = params['env']
        self.env_params['state_transform'] = state_transform
        self.env = RunEnv2(**self.env_params)

    def initialize(self, config, weights):
        self.create_actor(config['model_params'])
        self.create_env(config['env_params'])
        self.set_actor_weights(weights)

    @property
    def initialized(self):
        return self.actor is not None and self.rand_process is not None and self.env is not None

    def _set_noisy_weights(self, weights_means, weights_sigmas):
        weights = []
        for mu, sigma in zip(weights_means, weights_sigmas):
            weights.append(np.random.normal(mu, sigma))
        self.actor.set_actor_weights(weights)

    def run_episode(self, weights_means, weights_sigmas, n_times=5):

        # set weights of actor
        self._set_noisy_weights(weights_means, weights_sigmas)

        start = time()

        total_reward = 0.
        total_reward_original = 0.
        steps = 0.

        for _ in range(n_times):
            seed = random.randrange(2 ** 32 - 2)
            state = self.env.reset(seed=seed, difficulty=2)
            terminal = False

            while not terminal:
                state = np.asarray(state, dtype='float32')
                action = self.actor.act(state)

                next_state, reward, next_terminal, info = self.env.step(action)
                total_reward += reward
                total_reward_original += info['original_reward']
                steps += 1

                state = next_state
                terminal = next_terminal

                if terminal:
                    break

            self.total_episodes += 1

            if self.total_episodes % 100 == 0:
                self.env = RunEnv2(**self.env_params)

        ret = {
            'total_reward': total_reward/n_times,
            'total_reward_original': total_reward_original/n_times,
            'steps': steps/n_times,
            'time_took': (time() - start)/60,
            'weights': [w.tolist() for w in self.actor.get_actor_weights()]
        }

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
