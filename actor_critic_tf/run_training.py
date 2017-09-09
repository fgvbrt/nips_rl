import os
os.environ['OMP_NUM_THREADS'] = '1'

import random
import argparse
import numpy as np
from model import ActorCritic, ActorCriticNumpy, PPO, PolicyGradient
import tensorflow as tf
from multiprocessing import Process, Value, cpu_count, Queue
from time import time
from utils import elu, discount
from environments import RunEnv2
from state import StateVelCentr
from test_env import make_test_env


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--num_workers', default=cpu_count(), type=int, help="Number of workers.")
    parser.add_argument('--gamma', type=float, default=0.95, help="Discount factor for reward.")
    parser.add_argument('--max_steps', type=int, default=100000000, help="Number of epochs.")
    parser.add_argument('--test_period', default=20, type=int, help="Period testing and saving weighs.")
    parser.add_argument('--num_test_episodes', type=int, default=3, help="Number of test episodes.")
    parser.add_argument('--epochs_per_train', type=int, default=1, help="Number of test episodes.")
    parser.add_argument('--distr', type=str, default='bernoulli', help="Distribution type.")
    parser.add_argument('--log_dir', type=str, default='log', help="log directory.")
    return parser.parse_args()


def trajectory_generator(ac, state_transform, data_queue, weights_queue,
                         weights_update_period, global_step,
                         best_test_reward):

    env = RunEnv2(state_transform, visualize=False)
    #env = make_test_env()

    # prepare buffers for data
    states = []
    actions = []
    rewards = []

    total_episodes = 0
    start = time()
    running = True
    while running:
        seed = random.randrange(2 ** 32 - 2)
        state = env.reset(seed=seed)
        #state = env.reset()

        total_reward = 0.
        terminal = False
        steps = 0
        while not terminal:
            state = np.asarray(state, dtype='float32')

            action = ac.act(state)

            next_state, reward, next_terminal, _ = env.step(action)
            total_reward += reward
            steps += 1
            global_step.value += 1

            # add data to buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            terminal = next_terminal

            if terminal:
                break

        total_episodes += 1

        data = (np .asarray(states).astype('float32'),
                np.asarray(actions).astype('float32'),
                np.asarray(rewards).astype('float32'),
                )
        # send data for training
        data_queue.put(data)

        # clear buffers
        del states[:]
        del actions[:]
        del rewards[:]

        # get and set weights
        if total_episodes % weights_update_period == 0:
            weights = weights_queue.get()
            if weights is None:
                running = False
            elif isinstance(weights, list):
                ac.set_weights(weights)

        # print statistics
        report_str = 'Global step: {}, steps/sec: {:.2f}, episode length {}, reward: {:.2f}, best reward: {:.2f}'. \
            format(global_step.value, 1. * global_step.value / (time() - start), steps, total_reward,
                   best_test_reward.value)
        print report_str


def test_agent(ac, state_transform, best_reward, num_test_episodes):
    # environment
    env = RunEnv2(state_transform, visualize=False)
    #env = make_test_env()

    total_reward = 0
    for ep in range(num_test_episodes):
        state = env.reset()
        while True:
            state = np.asarray(state, dtype='float32')
            action = ac.act(state, False)
            state, reward, terminal, _ = env.step(action)
            total_reward += reward
            if terminal:
                break

    mean_reward = 1. * total_reward / num_test_episodes
    print('test reward {:.2f}'.format(mean_reward))
    if mean_reward > best_reward.value:
        best_reward.value = mean_reward
        ac.save_weights('weights/weights_reward_{}.pkl'.format(int(mean_reward)))


def run_training():
    args = get_args()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # agent
    state_transform = StateVelCentr(last_n_bodies=0, exclude_centr=True)

    ac = ActorCritic(state_transform.state_size, 18, args.distr)
    ac_old = ActorCritic(state_transform.state_size, 18, args.distr, scope='actor_critic_old')
    ppo = PPO(ac, ac_old)
    #pg = PolicyGradient(ac)
    sess.run(tf.global_variables_initializer())
    weights = sess.run(ac.actor_params)
    ac_np = ActorCriticNumpy(weights, elu, args.distr)
    ppo.update_old_ac(sess)

    train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    # init shared variables
    global_step = Value('i', 0)
    best_reward = Value('f', -1e8)

    # init workers
    workers = []
    weights_queues = []
    data_queues = []
    for _ in xrange(8):
        data_queue = Queue()
        weights_queue = Queue()
        w = Process(target=trajectory_generator,
                    args=(ac_np, state_transform, data_queue,
                          weights_queue, args.epochs_per_train,
                          global_step, best_reward))
        w.daemon = True
        w.start()
        workers.append(w)
        weights_queues.append(weights_queue)
        data_queues.append(data_queue)

    epoch = 0
    running = True
    S = []
    A = []
    T = []
    p = None
    while running:
        # get data from workers
        for _ in xrange(args.epochs_per_train):
            for data_queue in data_queues:
                states, actions, rewards = data_queue.get()
                S.append(states)
                A.append(actions)
                T.append(discount(rewards, args.gamma))

        states = np.concatenate(S)
        actions = np.concatenate(A)
        targets = np.concatenate(T)

        del S[:]
        del A[:]
        del T[:]

        # training
        ppo.train(states, actions, targets, 10, sess, train_writer)
        #pg.train(states, actions, targets, sess, train_writer)

        weights = sess.run(ac.actor_params)
        if global_step.value > args.max_steps:
            running = False
            weights = None

        for w_queue in weights_queues:
            w_queue.put(weights)

        epoch += 1

        # run testing process
        if epoch % args.test_period == 0 or weights is None:
            weights = sess.run(ac.actor_params)
            ac_np.set_weights(weights)
            p = Process(target=test_agent,
                        args=(ac_np, state_transform, best_reward, args.num_test_episodes)
                        )
            p.daemon = True
            p.start()

    for w in workers:
        w.join()

    if p is not None:
        p.join()

    sess.close()


if __name__ == '__main__':
    run_training()
