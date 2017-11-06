import os
os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'

import argparse
import numpy as np
from model import build_model, Agent
from time import sleep
from multiprocessing import Process, Value, Queue
import queue
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


def find_samplers():
    samplers = []
    with Pyro4.locateNS() as ns:
        for sampler, sampler_uri in ns.list(prefix="sampler.").items():
            print("found sampler", sampler)
            samplers.append(Pyro4.Proxy(sampler_uri))
    if not samplers:
        raise ValueError("no samplers found! (have you started the samplers first?)")
    return samplers


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
    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(**config['model_params'])
    actor = Agent(actor_fn, params_actor, params_crit)
    if args.weights is not None:
        actor.load(args.weights)
    weights = [w.tolist() for w in actor.get_actor_weights()]

    # initialize samplers
    samplers = find_samplers()
    init_samplers(samplers, config, weights)

    # init replay memory
    memory = ReplayMemory(state_transform.state_size, 18, **config['repay_memory'])

    # learning rate decay step
    def get_lr_step(lr, lr_end, max_steps):
        return (lr - lr_end) / max_steps

    actor_lr_step = get_lr_step(
        config['model_params']['actor_lr'],
        config['train_params']['actor_lr_end'],
        config['train_params']['max_steps']
    )
    critic_lr_step = get_lr_step(
        config['model_params']['critic_lr'],
        config['train_params']['critic_lr_end'],
        config['train_params']['max_steps']
    )


    # check sampling


    # start sampling
    samplers_results = {s: Pyro4.Future(s.run_episode)() for s in samplers}

    # common statistic
    total_steps = 0
    prev_steps = 0
    updates = 0
    best_reward = -1e8
    weights_from_samplers = []
    start = time()
    start_save = start
    # main train loop
    while total_steps < config['train_params']['max_steps']:
        for s, res in samplers_results.items():
            if res.ready:
                res = res.value
                total_steps += res['steps']

                # start new job
                new_res = process_results(s, res, memory, weights, weights_from_samplers)
                samplers_results[s] = new_res

                # report progress on this episode
                report_str = 'Global step: {}, steps/sec: {:.2f}, updates: {}, episode len {}, ' \
                             'reward: {:.2f}, original_reward {:.4f}; best reward: {:.2f} noise {}'. \
                    format(total_steps, 1. * total_steps / (time() - start), updates,
                           res['steps'], res['total_reward'], res['total_reward_original'],
                           best_reward, 'actions' if res['action_noise'] else 'params')
                print(report_str)

        # check if enough samles and can start training
        if len(memory) > config['train_params']['start_train_steps']:
            # batch2 if faster than batch1
            batch = memory.random_batch2(args.batch_size)

            # flip states
            if np.random.rand() < args.flip_prob:
                states, actions, rewards, terminals, next_states = batch

                states_flip = state_transform.flip_states(states)
                next_states_flip = state_transform.flip_states(next_states)
                actions_flip = np.zeros_like(actions)
                actions_flip[:, :9] = actions[:, 9:]
                actions_flip[:, 9:] = actions[:, :9]

                states_all = np.concatenate((states, states_flip))
                actions_all = np.concatenate((actions, actions_flip))
                rewards_all = np.tile(rewards.ravel(), 2).reshape(-1, 1)
                terminals_all = np.tile(terminals.ravel(), 2).reshape(-1, 1)
                next_states_all = np.concatenate((next_states, next_states_flip))
                batch = (states_all, actions_all, rewards_all, terminals_all, next_states_all)

            actor_loss, critic_loss = train_fn(*batch)
            updates += 1
            if np.isnan(actor_loss):
                raise Value('actor loss is nan')
            if np.isnan(critic_loss):
                raise Value('critic loss is nan')
            target_update_fn()
            weights = actor.get_actor_weights()

            delta_steps = total_steps - prev_steps
            prev_steps += delta_steps

            actor_lr.set_value(lasagne.utils.floatX(
                max(actor_lr.get_value() - delta_steps*actor_lr_step, args.actor_lr_end)))
            critic_lr.set_value(lasagne.utils.floatX(
                max(critic_lr.get_value() - delta_steps*critic_lr_step, args.critic_lr_end)))

        # check if need to save and test
        if (time() - start_save)/60. > config['test_params']['save_period_min']:
            fname = os.path.join(save_dir, 'weights_updates_{}.h5'.format(updates.value))
            actor.save(fname)
            start_save = time()

        #  start new test process
    #     weights_rew_to_check = [(w, r) for w, r in weights_rew_to_check if r > best_reward.value]
    #     if ((time() - start_test) / 60. > args.test_period_min or len(weights_rew_to_check) > 0) and testing.value == 0:
    #         testing.value = 1
    #         print('start test')
    #         if len(weights_rew_to_check) > 0:
    #             _weights, _ = weights_rew_to_check.pop()
    #         else:
    #             _weights = weights
    #         worker = Process(target=test_agent,
    #                          args=(testing, state_transform, args.num_test_episodes,
    #                                model_params, _weights, best_reward, updates, save_dir)
    #                          )
    #         worker.daemon = True
    #         worker.start()
    #         start_test = time()
    #
    # # end all processes
    # for w in workers:
    #     w.join()


if __name__ == '__main__':
    main()
