import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['THEANO_FLAGS'] = 'device=cpu'
import Pyro4
import numpy as np
from model import build_model, Agent
from environments import RunEnv2
from state import StateVelCentr
import random
import argparse


@Pyro4.expose
class Tester(object):
    def __init__(self):
        self.num_test_episodes = None
        self.actor = None
        self.env_params = None
        self.env = None
        self.save_dir = None

    def create_actor(self, params):
        _, _, _, actor_fn, _, params_actor, params_crit, _, _ = \
            build_model(**params)
        self.actor = Agent(actor_fn, params_actor, params_crit)

    def set_actor_weights(self, weights):
        self.actor.set_weights(*weights)

    def create_env(self, params):
        state_transform = StateVelCentr(**params['state_transform'])
        self.env_params = params['env']
        # for test we will make 1 step
        self.env_params['skip_frame'] = 1
        self.env_params['state_transform'] = state_transform
        self.env = RunEnv2(**self.env_params)

    def initialize(self, config, weights):
        self.num_test_episodes = config['test_params']['num_test_episodes']
        self.save_dir = config['test_params']['save_dir']

        self.create_actor(config['model_params'])
        self.create_env(config['env_params'])
        self.set_actor_weights(weights)

    def test_model(self, weights, best_reward):
        # set weights of actor
        self.actor.set_weights(*weights)

        # init new env
        env = RunEnv2(**self.env_params)

        # start testing
        test_rewards = []
        print('\nstart testing')
        for ep in range(self.num_test_episodes):
            seed = random.randrange(2 ** 32 - 2)
            state = env.reset(seed=seed, difficulty=2)
            test_reward = 0
            while True:
                state = np.asarray(state, dtype='float32')
                action = self.actor.act(state)
                state, reward, terminal, _ = env.step(action)
                test_reward += reward
                if terminal:
                    break

            print('test reward {:.2f}'.format(test_reward))
            test_rewards.append(test_reward)

        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)

        print('test reward mean: {:.2f}, std: {:.2f}'.format(float(mean_reward), float(std_reward)))

        ret_weights = None
        if mean_reward > best_reward or mean_reward > 30 * env.reward_mult:
            fname = os.path.join(self.save_dir, 'weights_reward_{:.2f}.h5'.format(mean_reward))
            self.actor.save(fname)
            ret_weights = [w.tolist() for w in self.actor.get_actor_weights()]

        return mean_reward, ret_weights


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--name', type=str, default='1', help="Name of server")
    parser.add_argument('--host', type=str, default='localhost', help="Host name.")
    parser.add_argument('--port', type=int, default=1234, help="Host port.")
    parser.add_argument('--nathost', type=str, default=None, help="Nat name.")
    parser.add_argument('--natport', type=int, default=None, help="Nat port.")
    parser.add_argument('--ns_host', type=str, default='192.168.1.130', help="Name server host.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    tester = Tester()

    # for example purposes we will access the daemon and name server ourselves and not use serveSimple
    with Pyro4.Daemon(host=args.host, port=args.port, nathost=args.nathost, natport=args.natport) as daemon:
        uri = daemon.register(tester)
        name = "tester.{}.port.{}".format(args.name, args.port)
        with Pyro4.locateNS(host=args.ns_host) as ns:
            ns.register(name, uri)
        print("Tester ready: name {} uri {}".format(name, uri))
        daemon.requestLoop()
