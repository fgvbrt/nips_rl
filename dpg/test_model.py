import argparse
from agent import ActorNumpy, elu
import cPickle
from osim.env import RunEnv
import numpy as np
from osim.http.client import Client

TOKEN = 'e298627d07b55655ef35bb859431813e'
REMOTE_BASE = 'http://grader.crowdai.org:1729'


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('model', type=str, help="File with model.")
    parser.add_argument('--num_test_episodes', type=int, default=3, help="Number of test episodes.")
    parser.add_argument('--submit', action='store_true')
    return parser.parse_args()


def submit_agent(agent):
    client = Client(REMOTE_BASE)
    s = client.env_create(TOKEN)
    all_total_rewards = []
    total_reward = 0
    while True:
        s = np.asarray(s, dtype='float32')
        a = agent.act(s)
        s, r, done, _ = client.env_step(a.tolist())
        total_reward += r
        if done:
            print(total_reward)
            all_total_rewards.append(total_reward)
            total_reward = 0
            s = client.env_reset()
            if not s:
                break

    client.submit()
    return all_total_rewards


def test_agent(agent, num_test_episodes=3):
    env = RunEnv(visualize=False)
    all_total_rewards = []
    for ep in xrange(num_test_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            state = np.asarray(state, dtype='float32')
            action = agent.act(state)
            state, reward, terminal, _ = env.step(action)
            total_reward += reward
            if terminal:
                break

        print total_reward
        all_total_rewards.append(total_reward)
    return all_total_rewards


def test_model():
    args = get_args()

    with open(args.model, 'rb') as f:
        weights = cPickle.load(f)

    actor = ActorNumpy(weights, elu)

    if args.submit:
        all_total_rewards = submit_agent(actor)
    else:
        all_total_rewards = test_agent(actor, args.num_test_episodes)

    print 'mean reward {:.2f}'.format(sum(all_total_rewards)/len(all_total_rewards))


if __name__ == '__main__':
    test_model()
