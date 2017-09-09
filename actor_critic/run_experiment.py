import os
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
from model import build_model
from time import sleep
from multiprocessing import Process, cpu_count, Value, Queue
from agent import Actor, run_agent, elu, sigmoid, RunEnv2


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for reward.")
    parser.add_argument('--num_agents', type=int, default=cpu_count()-1, help="Number of agents to run.")
    parser.add_argument('--sleep', type=int, default=1, help="Sleep time in seconds before start each worker.")
    parser.add_argument('--max_steps', type=int, default=100000000, help="Number of steps.")
    parser.add_argument('--testing_period', default=20, type=int, help="Period testing and saving weighs.")
    parser.add_argument('--num_test_episodes', type=int, default=3, help="Number of test episodes.")
    parser.add_argument('--batch_episodes', type=int, default=5, help="Number of episodes for each agent to form batch.")
    return parser.parse_args()


def main():
    args = get_args()

    # build env for testing
    env = RunEnv2(visualize=False)

    # build model
    train_fn, actor_fn, value_fn, params_actor = build_model(41, 18)

    # build actor
    weights = [p.get_value() for p in params_actor]
    activation = np.tanh
    actor = Actor(weights, activation)

    # init shared variables
    global_step = Value('i', 0)
    best_reward = Value('f', -1e8)

    # init agents

    workers = []
    weights_queues = []
    data_queues = []
    for i in xrange(args.num_agents):
        w_queue = Queue()
        data_queue = Queue()
        worker = Process(target=run_agent,
                         args=(actor, data_queue, w_queue, args.gamma,
                               args.batch_episodes, global_step, best_reward,
                               args.max_steps)
                         )
        worker.daemon = True
        worker.start()
        sleep(args.sleep)
        workers.append(worker)
        weights_queues.append(w_queue)
        data_queues.append(data_queue)

    epoch = 0
    while global_step.value < args.max_steps:

        # get data from workers
        S = []
        A = []
        T = []
        for data_queue in data_queues:
            states, actions, targets = data_queue.get()
            S.append(states)
            A.append(actions)
            T.append(targets)

        S = np.concatenate(S)
        A = np.concatenate(A)
        T = np.concatenate(T)
        train_fn(S, A, T)

        weights = [p.get_value() for p in params_actor]
        for w_queue in weights_queues:
            w_queue.put(weights)

        epoch += 1

        if epoch % args.testing_period == 0:
            total_reward = 0
            actor.set_weights(weights)

            for ep in range(args.num_test_episodes):
                state = env.reset()
                while True:
                    state = np.asarray(state, dtype='float32')
                    action = actor.choose_action(state, False)
                    state, reward, terminal, _ = env.step(action)
                    total_reward += reward
                    if terminal:
                        break

            mean_reward = 1. * total_reward / args.num_test_episodes
            print 'test reward mean', mean_reward
            if mean_reward > best_reward.value:
                best_reward.value = mean_reward
                fname = 'weights_steps_{}_reward_{}.pkl'.format(global_step.value, int(mean_reward))
                actor.save_weights(fname)

    # end all processes
    for w in workers:
        w.join()

if __name__ == '__main__':
    main()
