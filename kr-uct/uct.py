import random
import numpy as np
from math import sqrt, log
from sklearn.metrics.pairwise import rbf_kernel
from collections import OrderedDict

MIN_NODE_ACTIONS = 10
GOOD_ACTIONS = 5
SIGMA1 = 0.1
SIGMA2 = 0.2
RBF_GAMMA = 10


def rand_argmax(lst):
    max_l = max(lst)
    best_is = [i for i, l in enumerate(lst) if l == max_l]
    i = random.randint(0, len(best_is)-1)
    return best_is[i]


def ucb(visits, value, parent_visits, exploration):
    return value + exploration * sqrt(log(parent_visits)/visits)


def moving_average(v, n, vi):
    return (n * v + vi) / (n + 1)


class Node(object):
    n_actions = 0

    def __init__(self, osim_state, state, terminal=0, reward=0, parent=None):
        self.osim_state = osim_state
        self.state = state
        self.terminal = terminal
        self.reward = reward  # immediate reward during simulation for calculating parent reward
        self.parent = parent  # parent Node
        self.childs = OrderedDict()  # dict of child nodes for each action, key is action, value  is Node

        self.value = 0.  # current value for node
        self.visits = 0  # current number of visits for node
        self.w_table = np.zeros(shape=(MIN_NODE_ACTIONS, MIN_NODE_ACTIONS))

    def _update_w_table(self, a):
        n = len(self.childs) - 1
        for i, b in enumerate(self.childs.keys()):
            rbf = rbf_kernel([a], [b], RBF_GAMMA)
            self.w_table[i, n] = rbf
            self.w_table[n, i] = rbf

    def best_action(self):
        visits = [c.visits for c in self.childs.values()]
        return list(self.childs.keys())[rand_argmax(visits)]

    def ucb_action(self, c):
        visits = np.asarray([c.visits for c in self.childs.values()])
        values = np.asarray([c.value for c in self.childs.values()])

        _w_vis = self.w_table * visits
        _w_vis_val = _w_vis * values

        w = np.sum(_w_vis, axis=1)
        e = _w_vis_val.sum(axis=1) / w
        ucb_vals = e + c*np.sqrt(np.log(w.sum()) / w)
        #print('\n')
        #print('ucb:', ucb_vals)
        #print('e:', e)
        #print('exploration:', np.sqrt(np.log(w.sum()) / w))
        a_idxs = rand_argmax(ucb_vals)

        return list(self.childs.keys())[a_idxs]

    def selection(self, env, agent, c=1):
        # restore env state
        env.restore_state(self.osim_state)

        # need to take some actions for exploration
        if len(self.childs) < MIN_NODE_ACTIONS:
            # it is sure leaf
            leaf = True

            # action according to policy
            if len(self.childs) == 0:
                a = agent.act(self.state)
            # action according to policy with some noise
            elif len(self.childs) < GOOD_ACTIONS:
                a = agent.act(self.state)
                a += np.random.normal(0, SIGMA1, 18)
            # just random action
            else:
                # TODO: more clever choice
                a = agent.act(self.state)
                a += np.random.normal(0, SIGMA2, 18)
                #a = np.random.rand(18)

            a = tuple(np.clip(a, 0, 1))

            # if leaf we need to make step
            state, r, t, _ = env.step(a)
            osim_state = env.clone_state()

            # create new node
            node = Node(osim_state, state, t, r, self)
            
            # add child to parent
            self.childs[a] = node

            # update w_table
            self._update_w_table(a)

        # all action have been taken, choose according to max q value + exploration
        else:
            leaf = False
            # choose ucb action
            a = self.ucb_action(c)
            node = self.childs[a]

        return node, leaf


def uct_action(env, agent, node, sim_steps, search_horizont, gamma, c=1.):
    
    # do simulations
    for _ in range(sim_steps):
        sample(env, agent, node, search_horizont, gamma, c)

    # choose action
    return node.best_action()


def sample(env, agent, node, search_horizont, gamma, c):
    
    depth = 0
    leaf = False

    # while not leaf and depth < search_horizont:
    while not leaf:
        node, leaf = node.selection(env, agent, c)
        depth += 1

        # break in terminal
        if node.terminal:
            break

    R = node.value
    if leaf and not node.terminal:
        env.restore_state(node.osim_state)
        s = node.state
        R = rollout(env, s, agent, search_horizont, gamma)

    # backup
    update_values(R, node, gamma)


def rollout(env, s, agent, n_steps, gamma=0.99):
    R = 0.
    g = 1.
    step = 0

    while True:
        a = agent.act(s)
        s, r, t, _, = env.step(a)
        R += r*g
        g *= gamma
        step += 1
        if t or 0 < n_steps <= step:
            break

    return R


def update_values(R, node, gamma=0.99):
    while node is not None:
        # update this node info
        node.value = moving_average(node.value, node.visits, R)
        node.visits += 1

        # calculate value for parent
        R = node.reward + gamma * R

        # make parent current node
        node = node.parent
