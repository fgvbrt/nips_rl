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


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--gamma', type=float, default=0.9, help="Discount factor for reward.")
    parser.add_argument('--max_steps', type=int, default=10000000, help="Number of steps.")
    parser.add_argument('--test_period_min', default=30, type=int, help="Test interval int min.")
    parser.add_argument('--save_period_min', default=30, type=int, help="Save interval int min.")
    parser.add_argument('--num_test_episodes', type=int, default=5, help="Number of test episodes.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size.")
    parser.add_argument('--start_train_steps', type=int, default=10000, help="Number of steps tp start training.")
    parser.add_argument('--critic_lr', type=float, default=2e-3, help="critic learning rate")
    parser.add_argument('--actor_lr', type=float, default=1e-3, help="actor learning rate.")
    parser.add_argument('--critic_lr_end', type=float, default=5e-5, help="critic learning rate")
    parser.add_argument('--actor_lr_end', type=float, default=5e-5, help="actor learning rate.")
    parser.add_argument('--flip_prob', type=float, default=1., help="Probability of flipping.")
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normaliation.")
    parser.add_argument('--exp_name', type=str, default=datetime.now().strftime("%d.%m.%Y-%H:%M"),
                        help='Experiment name')
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    return parser.parse_args()


def init_model(args, state_transform):
    # build model
    model_params = {
        'state_size': state_transform.state_size,
        'num_act': 18,
        'gamma': args.gamma,
        'actor_lr': args.actor_lr,
        'critic_lr': args.critic_lr,
        'layer_norm': args.layer_norm
    }
    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(**model_params)
    actor = Agent(actor_fn, params_actor, params_crit)

    if args.weights is not None:
        actor.load(args.weights)

    return actor, train_fn


def find_samplers():
    samplers = []
    with Pyro4.locateNS() as ns:
        for sampler, sampler_uri in ns.list(prefix="sampler.").items():
            print("found sampler", sampler)
            samplers.append(Pyro4.Proxy(sampler_uri))
    if not samplers:
        raise ValueError("no samplers found! (have you started the samplers first?)")
    return samplers


def init_samplers(samplers):
    pass


def main():
    args = get_args()

    # create save directory
    save_dir = os.path.join('weights', args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # init state transform
    state_transform = StateVelCentr(obstacles_mode='standard',
                                    exclude_centr=True)
    # init model
    actor, train_fn = init_model(args, state_transform)

    # init replay memory
    memory = ReplayMemory(state_transform.state_size, 18, 5000000)

    # find samplers
    samplers = find_samplers()



    actor_lr_step = (args.actor_lr - args.actor_lr_end) / args.max_steps
    critic_lr_step = (args.critic_lr - args.critic_lr_end) / args.max_steps

    #
    #
    # # init agents
    # data_queue = Queue()
    # workers = []
    # weights_queues = []
    # print('starting {} agents'.format(args.num_agents))
    # for i in range(args.num_agents):
    #     w_queue = Queue()
    #     worker = Process(target=run_agent,
    #                      args=(model_params, weights, state_transform, data_queue, w_queue,
    #                            i, global_step, updates, best_reward, args.max_steps)
    #                      )
    #     worker.daemon = True
    #     worker.start()
    #     sleep(args.sleep)
    #     workers.append(worker)
    #     weights_queues.append(w_queue)
    #
    # prev_steps = 0
    # start_save = time()
    # start_test = time()
    # weights_rew_to_check = []
    # while global_step.value < args.max_steps:
    #
    #     # get all data
    #     try:
    #         i, batch, weights_check, reward = data_queue.get_nowait()
    #         if weights_check is not None:
    #             weights_rew_to_check.append((weights_check, reward))
    #         weights_queues[i].put(weights)
    #         # add data to memory
    #         memory.add_samples(*batch)
    #     except queue.Empty:
    #         pass
    #
    #     # training step
    #     # TODO: consider not training during testing model
    #     if len(memory) > args.start_train_steps:
    #         batch = memory.random_batch2(args.batch_size)
    #
    #         if np.random.rand() < args.flip_prob:
    #             states, actions, rewards, terminals, next_states = batch
    #
    #             states_flip = state_transform.flip_states(states)
    #             next_states_flip = state_transform.flip_states(next_states)
    #             actions_flip = np.zeros_like(actions)
    #             actions_flip[:, :num_actions//2] = actions[:, num_actions//2:]
    #             actions_flip[:, num_actions//2:] = actions[:, :num_actions//2]
    #
    #             states_all = np.concatenate((states, states_flip))
    #             actions_all = np.concatenate((actions, actions_flip))
    #             rewards_all = np.tile(rewards.ravel(), 2).reshape(-1, 1)
    #             terminals_all = np.tile(terminals.ravel(), 2).reshape(-1, 1)
    #             next_states_all = np.concatenate((next_states, next_states_flip))
    #             batch = (states_all, actions_all, rewards_all, terminals_all, next_states_all)
    #
    #         actor_loss, critic_loss = train_fn(*batch)
    #         updates.value += 1
    #         if np.isnan(actor_loss):
    #             raise Value('actor loss is nan')
    #         if np.isnan(critic_loss):
    #             raise Value('critic loss is nan')
    #         target_update_fn()
    #         weights = actor.get_actor_weights()
    #
    #     delta_steps = global_step.value - prev_steps
    #     prev_steps += delta_steps
    #
    #     actor_lr.set_value(lasagne.utils.floatX(max(actor_lr.get_value() - delta_steps*actor_lr_step, args.actor_lr_end)))
    #     critic_lr.set_value(lasagne.utils.floatX(max(critic_lr.get_value() - delta_steps*critic_lr_step, args.critic_lr_end)))
    #
    #     # check if need to save and test
    #     if (time() - start_save)/60. > args.save_period_min:
    #         fname = os.path.join(save_dir, 'weights_updates_{}.h5'.format(updates.value))
    #         actor.save(fname)
    #         start_save = time()
    #
    #     # start new test process
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
