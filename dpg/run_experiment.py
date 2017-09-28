import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'

import argparse
import numpy as np
from model import build_model, Agent
from time import sleep
from multiprocessing import Process, cpu_count, Value, Queue
import Queue as queue
from memory import ReplayMemory
from agent import run_agent, elu, sigmoid
from state import StateVelCentr, StateVel
import lasagne


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--gamma', type=float, default=0.995, help="Discount factor for reward.")
    parser.add_argument('--num_agents', type=int, default=cpu_count()-1, help="Number of agents to run.")
    parser.add_argument('--sleep', type=int, default=0, help="Sleep time in seconds before start each worker.")
    parser.add_argument('--max_steps', type=int, default=10000000, help="Number of steps.")
    parser.add_argument('--test_period', default=10, type=int, help="Period testing and saving weighs.")
    parser.add_argument('--num_test_episodes', type=int, default=10, help="Number of test episodes.")
    parser.add_argument('--batch_size', type=int, default=2000, help="Batch size.")
    parser.add_argument('--start_train_steps', type=int, default=10000, help="Number of steps tp start training.")
    parser.add_argument('--critic_lr', type=float, default=2e-3, help="critic learning rate")
    parser.add_argument('--actor_lr', type=float, default=1e-3, help="actor learning rate.")
    parser.add_argument('--critic_lr_end', type=float, default=5e-5, help="critic learning rate")
    parser.add_argument('--actor_lr_end', type=float, default=5e-5, help="actor learning rate.")
    parser.add_argument('--flip_prob', type=float, default=0., help="Probability of flipping.")
    return parser.parse_args()


def main():
    args = get_args()

    #state_transform = StateVelCentr(exclude_obstacles=True)
    state_transform = StateVelCentr(obstacles_mode='standard', exclude_centr=True, vel_states=[])
    #state_transform = StateVel(exclude_obstacles=True)
    num_actions = 18

    # build model
    model_params = [state_transform.state_size, num_actions, args.gamma, args.actor_lr, args.critic_lr]
    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(*model_params)

    actor_lr_step = (args.actor_lr - args.actor_lr_end) / args.max_steps
    critic_lr_step = (args.critic_lr - args.critic_lr_end) / args.max_steps

    # build actor
    weights = [p.get_value() for p in params_actor]
    actor = Agent(actor_fn, params_actor, params_crit)

    # build replay memory
    memory = ReplayMemory(state_transform.state_size, 18, 5000000)

    # init shared variables
    global_step = Value('i', 0)
    updates = Value('i', 0)
    best_reward = Value('f', -1e8)

    # init agents
    data_queue = Queue()
    workers = []
    weights_queues = []
    for i in xrange(args.num_agents):
        w_queue = Queue()
        worker = Process(target=run_agent,
                         args=(model_params, weights, state_transform, data_queue, w_queue,
                               i, global_step, updates, best_reward, args.test_period,
                               args.num_test_episodes, args.max_steps)
                         )
        worker.daemon = True
        worker.start()
        sleep(args.sleep)
        workers.append(worker)
        weights_queues.append(w_queue)

    prev_steps = 0
    while global_step.value < args.max_steps:

        # get all data
        try:
            i, (states, actions, rewards, terminals) = data_queue.get_nowait()
            weights_queues[i].put(weights)
            # add data to memory
            memory.add_samples(states, actions, rewards, terminals)
        except queue.Empty:
            pass

        # training step
        if len(memory) > args.start_train_steps:
            batch = memory.random_batch(args.batch_size)

            if np.random.rand() < args.flip_prob:
                states, actions, rewards, terminals, next_states = batch

                states_flip = state_transform.flip_states(states)
                next_states_flip = state_transform.flip_states(next_states)
                actions_flip = np.zeros_like(actions)
                actions_flip[:, :num_actions//2] = actions[:, num_actions//2:]
                actions_flip[:, num_actions//2:] = actions[:, :num_actions//2]

                states_all = np.concatenate((states, states_flip))
                actions_all = np.concatenate((actions, actions_flip))
                rewards_all = np.tile(rewards.ravel(), 2).reshape(-1, 1)
                terminals_all = np.tile(terminals.ravel(), 2).reshape(-1, 1)
                next_states_all = np.concatenate((next_states, next_states_flip))
                batch = (states_all, actions_all, rewards_all, terminals_all, next_states_all)

            actor_loss, critic_loss = train_fn(*batch)
            updates.value += 1
            if np.isnan(actor_loss):
                raise Value('actor loss is nan')
            if np.isnan(critic_loss):
                raise Value('critic loss is nan')
            target_update_fn()
            weights = [p.get_value() for p in params_actor]

        delta_steps = global_step.value - prev_steps
        prev_steps += delta_steps

        actor_lr.set_value(lasagne.utils.floatX(max(actor_lr.get_value() - delta_steps*actor_lr_step, args.actor_lr_end)))
        critic_lr.set_value(lasagne.utils.floatX(max(critic_lr.get_value() - delta_steps*critic_lr_step, args.critic_lr_end)))

    # end all processes
    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
