import argparse
import numpy as np
from model import build_model
from time import sleep
from multiprocessing import Process, cpu_count, Value, Queue
import gym
from agent import Actor, run_agent, elu, sigmoid, gaussian, discount
import sklearn
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for reward.")
    parser.add_argument('--num_agents', type=int, default=cpu_count()-1, help="Number of agents to run.")
    parser.add_argument('--sleep', type=int, default=1, help="Sleep time in seconds before start each worker.")
    parser.add_argument('--max_steps', type=int, default=100000000, help="Number of steps.")
    parser.add_argument('--testing_period', default=5, type=int, help="Period testing and saving weighs.")
    parser.add_argument('--num_test_episodes', type=int, default=1, help="Number of test episodes.")
    parser.add_argument('--batch_episodes', type=int, default=10, help="Number of episodes for each agent to form batch.")
    return parser.parse_args()


def main():
    args = get_args()

    # build env for testing
    env = gym.envs.make("MountainCarContinuous-v0")

    # build feature former
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)
    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
            ])
    featurizer.fit(scaler.transform(observation_examples))
    pipeline = Pipeline([('skaler', scaler), ('rbf', featurizer)])

    # build model
    train_fn, actor_fn, value_fn, params_actor = build_model(400, env.action_space.shape[0])

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
                         args=(actor, pipeline, data_queue, w_queue, args.gamma,
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
                    state = pipeline.transform([state])[0]
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

    '''
    states = []
    actions = []
    rewards = []
    S = []
    A = []
    T = []
    total_steps = 0
    for i in xrange(args.max_steps):

        for bathch_ep in xrange(args.batch_episodes):
            steps = 0
            state = env.reset()
            total_reward = 0

            while True:
                state = pipeline.transform([state])[0]
                state = np.asarray(state, dtype='float32')
                mu, sigma = actor_fn([state])
                a = gaussian(mu, sigma, True)
                a = np.clip(a, -1, 1)[0]

                next_state, reward, next_terminal, _ = env.step(a)
                total_reward += reward
                total_steps += 1
                steps += 1

                states.append(state)
                actions.append(a)
                rewards.append(reward)

                state = next_state
                terminal = next_terminal

                if terminal:
                    print total_reward, steps
                    #print params_actor[-1].get_value()
                    break

            # add data to batch buffers
            S.append(np.asarray(states).astype('float32'))
            A.append(np.asarray(actions).astype('float32'))
            T.append(np.asarray(discount(rewards, args.gamma)).astype('float32'))

            # clear episode buffers
            del states[:]
            del actions[:]
            del rewards[:]

        train_fn(np.concatenate(S), np.concatenate(A), np.concatenate(T))

        # clear batch buffers
        del S[:]
        del A[:]
        del T[:]
    '''

if __name__ == '__main__':
    main()
