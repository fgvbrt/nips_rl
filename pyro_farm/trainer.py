import os
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
import numpy as np
from model import build_model, Agent
from memory import ReplayMemory
from state import StateVelCentr
import lasagne
from datetime import datetime
from time import time
import Pyro4
import yaml
import sys

sys.excepthook = Pyro4.util.excepthook


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--exp_name', type=str, default=datetime.now().strftime("%d.%m.%Y-%H:%M"),
                        help='Experiment name')
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    return parser.parse_args()


def find_workers(prefix):
    workers = []
    with Pyro4.locateNS() as ns:
        for sampler, sampler_uri in ns.list(prefix="{}.".format(prefix)).items():
            print("found {}".format(prefix), sampler)
            workers.append(Pyro4.Proxy(sampler_uri))
    if not workers:
        raise ValueError("no {} found!".format(prefix))
    print('found total {} {}s'.format(len(workers), prefix))
    return workers


def init_samplers(samplers, config, weights):
    results = []
    print('start samplers initialization')
    for sampler in samplers:
        res = Pyro4.Future(sampler.initialize)(config, weights)
        results.append(res)

    while len(results) > 0:
        for res in results:
            if res.ready:
                results.remove(res)
    print('finish samplers initialization')


def process_results(sampler, res, memory, weights, weights_from_samplers):
    # first set new weights
    sampler.set_actor_weights(weights)

    # start sampling process
    new_res = Pyro4.Future(sampler.run_episode)()

    # add data to memory
    memory.add_samples(res['states'], res['actions'], res['rewards'], res['terminals'])

    # add weights for tester check
    if 'weights' in res:
        weights_from_samplers.append((res['weights'], res['total_reward']))

    return new_res


def main():
    args = get_args()

    # create save directory
    save_dir = os.path.join('weights', args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read config
    with open('config.yaml') as f:
        config = yaml.load(f)

    # save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # init state transform
    state_transform = StateVelCentr(**config['env_params']['state_transform'])

    # init model
    config['model_params']['state_size'] = state_transform.state_size
    train_fn, train_actor, train_critic, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(**config['model_params'])
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.summary()

    if args.weights is not None:
        actor.load(args.weights)
    else:
        Warning('you started experiment without weights')

    # this is means of weights
    weights_mu = [w.tolist() for w in actor.get_actor_weights()]
    weights_sigma = [(np.ones_like(w)*config['train_params']['init_sigma']).tolist()
                     for w in weights_mu]

    # initialize samplers
    samplers = find_workers('sampler')
    init_samplers(samplers, config, weights_mu)

    # learning rate decay step
    def get_linear_decay_step(start, end, steps):
        return (start - end) / steps

    sigma = config['train_params']['sigma_constr_start']
    sigma_end = config['train_params']['sigma_constr_end']
    sigma_constr_step = get_linear_decay_step(
        config['train_params']['sigma_constr_start'],
        config['train_params']['sigma_constr_end'],
        config['train_params']['sigma_constr_steps']
    )

    # start sampling
    n_runs = config['train_params']['n_runs']
    samplers_results = {
        s: Pyro4.Future(s.run_episode)(weights_mu, weights_sigma, n_runs) for s in samplers
    }

    # common statistic
    episodes = 0
    best_reward = -1e8
    # main train loop
    print('start train loop')
    samplers_returns = []
    while episodes < config['train_params']['num_episodes']:

        # preocess results from samplers
        for s, res in samplers_results.items():
            if res.ready:
                res = res.value
                samplers_returns.append((res['weights'], res['total_reward']))

                # start new job
                new_res = Pyro4.Future(s.run_episode)(weights_mu, weights_sigma, n_runs)
                samplers_results[s] = new_res

                # report progress on this episode
                report_str = 'mean reward: {}, mean steps: {:.2f}, time: {:.2f}'.\
                    format(res['total_reward'], res['steps'], res['time_took'])
                print(report_str)

        # check if episode done
        if len(samplers_returns) >= config['train_params']['episode_runs']:
            episodes += 1
            samplers_returns = sorted(samplers_returns, key=lambda x: x[1], reverse=True)
            all_rewards = [r[1] for r in samplers_returns]
            all_weights = [r[0] for r in samplers_returns]

            top_n = int(len(samplers_returns)*config['train_params']['best_ratio'])
            mean_rew = float(np.mean(all_rewards))
            best_rew = all_rewards[0]
            top_n_rew = float(np.mean(all_rewards[:top_n]))

            print('episode: {}; mean reward: {:.2f}; top_n reward {:.2f}; best reward: {:.2f}'.\
                  format(episodes, mean_rew, top_n_rew, best_rew))

            # calculate new sigma and mu
            weights_mu = []
            weights_sigma = []
            for w in zip(*all_weights[:top_n]):
                weights_mu.append(np.mean(w, axis=0))
                weights_sigma.append(np.std(w, axis=0) + sigma)

            weights_mu = [w.tolist() for w in weights_mu]
            weights_sigma = [w.tolist() for w in weights_sigma]

            del samplers_returns[:]
            sigma = max(sigma_end, sigma-sigma_constr_step)

            # save top weights
            for i, w in enumerate(all_weights[:top_n]):
                actor.set_actor_weights(w)
                fname = os.path.join(save_dir, 'episode_{}_rew_{:.2f}.h5'.\
                                     format(episodes, all_rewards[i]))
                actor.save(fname)


if __name__ == '__main__':
    main()
