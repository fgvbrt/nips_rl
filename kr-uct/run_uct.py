import uct
import argparse
import os
from time import time
import numpy as np

parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=str, default="pong", help="Environment name.")
parser.add_argument('--version', type=str, default="v0", help="Version of environment.")
parser.add_argument('--act_rep', type=int, default=1, help="How many times repeat choosen action.")
parser.add_argument('--max_steps', type=int, default=10000, help="Maximum number of steps in environment.")
parser.add_argument('--sim_steps', default=100, type=int,
                    help="Number of simulations for selecting action with rollout policy.")
parser.add_argument('--search_horizont', default=100, type=int, help="Search_horizont for each simulation.")
parser.add_argument('--gamma', type=float, default=1., help="Discount factor for reward.")
parser.add_argument('--exploration', type=float, default=-2,
                    help="Coefficient of exploration part in action selecting during simulation.")
parser.add_argument('--prune_tree', action='store_true',
                    help="After choosing action with uct make tree pruning.\n"
                         "This means save tree and all visits for selecting new action from new state."
                         "Otherwise create new tree for selecting next new action.")
parser.add_argument('--rollout_agent_name', type=str, default=None,
                    help="Name of agent for rollouts: random or keras model filename.")
parser.add_argument('--behavior_agent_name', type=str, default=None,
                    help="Name of agent for behavior: random, keras model filename or 'uct'.")
parser.add_argument('--eps_greedy', type=float, default=0., help="Probability of selecting random action.")
parser.add_argument('--report_freq', type=int, default=100, help="Frequency of reporting uct progress.")


def run(env_name, version, act_rep, max_steps, rollout_agent_name,
        behavior_agent_name, eps_greedy, sim_steps, search_horizont,
        gamma=1., exploration=1., prune_tree=False, report_freq=100,
        n_runs=1, save_dir=None, save_freq=10, process=0):
    def save_data():
        if save_dir is not None and len(frames) > 0:
            run_data = {
                'frames': frames,
                'actions': actions,
                'reward': total_reward,
                'action_visits': action_visits,
                'action_values': action_values,
                'rewards': rewards,
                'action_meanings': env.env.get_action_meanings(),
            }
            fname = os.path.join(save_dir, 'run_process_{}_run_{}_steps_{}.pkl'.format(process, n_run, step))
            with open(fname, 'wb') as f:
                cPickle.dump(run_data, f, -1)

            del actions[:]
            del frames[:]
            del action_visits[:]
            del action_values[:]
            del rewards[:]

    env = create_env(env_name, version, act_rep)
    uct.Node.n_actions = env.action_space.n

    # agent for rollouts
    if rollout_agent_name == 'random' or rollout_agent_name is None:
        rollout_agent = RandomAgent(env.action_space.n)
    else:
        rollout_agent = KerasAgent(rollout_agent_name)

    # agent for action selections
    if behavior_agent_name == 'random':
        behavior_agent = RandomAgent(env.action_space.n)
    elif behavior_agent_name == 'uct' or behavior_agent_name is None:
        behavior_agent = 'uct'
    else:
        behavior_agent = KerasAgent(behavior_agent_name)

    if save_dir is not None:
        actions = []
        frames = []
        action_visits = []
        action_values = []
        rewards = []

    for n_run in xrange(n_runs):
        terminal = False

        env.reset()
        _frame = env.env._get_image()

        node = uct.Node(env.clone_state())

        total_reward = 0
        step = 0
        t_start = t0 = time()
        while not terminal:
            # choose uct action
            a_uct = uct.uct_action(env, rollout_agent, node, sim_steps, search_horizont, gamma, exploration)

            # choose action in environment
            if np.random.rand() < eps_greedy:
                a = env.action_space.sample()
            elif behavior_agent == 'uct':
                a = a_uct
            else:
                a = behavior_agent.choose_action(_frame)

            if save_dir is not None:
                actions.append(a_uct)
                frames.append(_frame)
                action_visits.append(node.a_visits)
                action_values.append(node.a_values)

            # do step in environment
            env.restore_state(node.state)
            frame, reward, terminal, _ = env.step(a)
            _frame = env.env._get_image()

            if save_dir is not None:
                rewards.append(reward)

            # create new tree or try to use old tree
            if prune_tree:
                if frame in node.childs[a]:
                    node = node.childs[a][frames]
                    node.parent = None
                else:
                    node = uct.Node(env.clone_state())
            else:
                node = uct.Node(env.clone_state())

            total_reward += reward
            step += 1

            # report progress
            if step % report_freq == 0:
                print
                'process: {} run: {}, steps: {}, time: {:.2f}, total reward: {:.2f}'. \
                    format(process, n_run + 1, step, time() - t0, total_reward)
                t0 = time()

            # save intermediate result
            if step % save_freq == 0:
                save_data()

            if 0 < max_steps < step:
                break

        print
        '\nprocess: {}, run: {}, total steps: {}, total time: {:.2f}, total reward: {:.2f}'. \
            format(process, n_run + 1, step, time() - t_start, total_reward)

        # save last chunk of data
        save_data()

    env.close()


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.env, args.version, args.act_rep, args.max_steps,
        args.rollout_agent_name, args.behavior_agent_name,
        args.eps_greedy, args.sim_steps, args.search_horizont,
        args.gamma, args.exploration, args.prune_tree, args.report_freq)
