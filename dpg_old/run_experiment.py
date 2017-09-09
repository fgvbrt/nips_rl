import numpy as np
from time import time
import argparse
from deep_learner import DeepLearner
from replay_memory import ReplayMemory
from random_process import OrnsteinUhlenbeckProcess
from osim.env import RunEnv


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for reward.")
    parser.add_argument('--max_steps', type=int, default=100000000, help="Number of steps.")
    parser.add_argument('--testing_period', default=10, type=int, help="Period testing and saving weighs.")
    parser.add_argument('--num_test_episodes', type=int, default=2, help="Number of test episodes.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size.")
    parser.add_argument('--train_freq', type=int, default=2, help="Train frequency.")
    parser.add_argument('--start_train_steps', type=int, default=2000, help="Number of steps tp start training.")
    return parser.parse_args()


def main():
    args = get_args()

    env = RunEnv(visualize=False)
    memory = ReplayMemory(41, 18)
    network = DeepLearner(memory, args.batch_size, args.gamma)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=18,
                                              sigma_min=1e-4, n_steps_annealing=2e6)

    total_steps = 0
    total_episodes = 0
    start = time()
    best_test_reward = -1e8
    while total_steps < args.max_steps:
        state = env.reset()
        random_process.reset_states()

        total_reward = 0.
        terminal = False
        steps = 0
        while not terminal:
            state = np.asarray(state, dtype='float32')

            action = network.choose_action(state)
            action += random_process.sample()

            next_state, reward, next_terminal, _ = env.step(action)
            total_reward += reward
            steps += 1
            total_steps += 1

            # add data to replay memory
            memory.add_sample(state, terminal, action, reward)
            state = next_state
            terminal = next_terminal

            # training
            if total_steps > args.start_train_steps and total_steps % args.train_freq == 0:
                batch = memory.random_batch(args.batch_size)
                network.train(*batch)

            if terminal:
                break

        # after episode finished add data to replay memory
        memory.add_sample(state, terminal, np.zeros(18), 0)

        total_episodes += 1

        if total_episodes % args.testing_period == 0:
            total_reward = 0
            for ep in range(args.num_test_episodes):
                state = env.reset()

                while True:
                    state = np.asarray(state, dtype='float32')
                    action = network.choose_action(state)
                    state, reward, terminal, _ = env.step(action)
                    total_reward += reward
                    if terminal:
                        break

            mean_reward = 1. * total_reward / args.num_test_episodes
            print 'test reward mean', mean_reward
            if mean_reward > best_test_reward:
                network.save_model('weights_steps_{}_reward_{}.pkl'.format(total_steps, int(mean_reward)))
                best_test_reward = mean_reward

        report_str = 'Global step: {}, steps/sec: {:.2f}, episode length {}, reward: {:.2f}, best reward: {:.2f}'. \
            format(total_steps, 1. * total_steps / (time() - start), steps, total_reward, best_test_reward)
        print report_str

        with open('report.log', 'a') as f:
            f.write(report_str + '\n')


if __name__ == '__main__':
    main()
