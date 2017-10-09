import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict
import cPickle
import numpy as np


def build_actor(l_input, num_act, last_nonlinearity=lasagne.nonlinearities.sigmoid,
                hid_sizes=(64, 64),
                nonlinearity=lasagne.nonlinearities.elu):
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(
            l_hid, hid_size,
            nonlinearity=nonlinearity
        )

    return lasagne.layers.DenseLayer(l_hid, num_act, nonlinearity=last_nonlinearity)


def build_critic(l_input, hid_sizes=(64, 32),
                 nonlinearity=lasagne.nonlinearities.elu):
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(
            l_hid, hid_size,
            nonlinearity=nonlinearity
        )

    return lasagne.layers.DenseLayer(l_hid, 1, nonlinearity=None)


def build_actor_critic(state_size, num_act):
    # input layers
    l_states = lasagne.layers.InputLayer([None, state_size])
    l_actions = lasagne.layers.InputLayer([None, num_act])
    l_input_critic = lasagne.layers.ConcatLayer([l_states, l_actions])
    # actor layer
    l_actor = build_actor(l_states, num_act)
    # critic layer
    l_critic = build_critic(l_input_critic)
    return l_states, l_actions, l_actor, l_critic


def build_model(state_size, num_act, gamma=0.99,
                actor_lr=0.00025,
                critic_lr=0.0005,
                target_update_coeff=1e-4,
                clip_delta=10.):

    # input tensors
    states = T.matrix('states')
    next_states = T.matrix('next_states')
    actions = T.matrix('actions')
    rewards = T.col('rewards')
    terminals = T.col('terminals')

    # current network
    l_states1, l_actions1, l_actor1, l_critic1 = build_actor_critic(state_size, num_act)
    l_states2, l_actions2, l_actor2, l_critic2 = build_actor_critic(state_size-3, num_act)
    # target network
    l_states_target1, l_actions_target1, l_actor_target1, l_critic_target1 =\
        build_actor_critic(state_size, num_act)
    l_states_target2, l_actions_target2, l_actor_target2, l_critic_target2 = \
        build_actor_critic(state_size-3, num_act)

    # get current network output tensors
    actions_pred1 = lasagne.layers.get_output(l_actor1, states)
    q_vals1 = lasagne.layers.get_output(l_critic1, {l_states1: states, l_actions1: actions})
    v_vals1 = lasagne.layers.get_output(l_critic1, {l_states1: states, l_actions1: actions_pred1})

    actions_pred2 = lasagne.layers.get_output(l_actor2, states[:, :-3])
    q_vals2 = lasagne.layers.get_output(l_critic2, {l_states2: states[:, :-3], l_actions2: actions})
    v_vals2 = lasagne.layers.get_output(l_critic2, {l_states2: states[:, :-3], l_actions2: actions_pred2})

    # get target network q-values
    actions_pred_target1 = lasagne.layers.get_output(l_actor_target1, next_states)
    v_vals_target1 = lasagne.layers.get_output(
        l_critic_target1,
        {l_states_target1: next_states, l_actions_target1: actions_pred_target1})

    actions_pred_target2 = lasagne.layers.get_output(l_actor_target2, next_states[:, :-3])
    v_vals_target2 = lasagne.layers.get_output(
        l_critic_target2,
        {l_states_target2: next_states[:, :-3], l_actions_target2: actions_pred_target2})

    # below are theano tensors
    m = T.eq(states[:, -3], -1).reshape((states.shape[0], 1))
    actions_pred = (1-m)*actions_pred1 + m*actions_pred2
    q_vals = (1-m)*q_vals1 + m*q_vals2
    v_vals = (1-m)*v_vals1 + m*v_vals2

    v_vals_target = (1-m)*v_vals_target1 + m*v_vals_target2

    # target for q_vals
    target = gamma*v_vals_target*(1.-terminals) + rewards
    td_error = target - q_vals

    # critic loss
    if clip_delta > 0:
        quadratic_part = T.minimum(abs(td_error), clip_delta)
        linear_part = abs(td_error) - quadratic_part
        critic_loss = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
    else:
        critic_loss = 0.5 * td_error ** 2
    critic_loss = T.mean(critic_loss)

    # actor loss
    actor_loss = -1.*T.mean(v_vals)

    # get params
    params_actor = lasagne.layers.get_all_params(l_actor1) + lasagne.layers.get_all_params(l_actor2)
    params_crit = lasagne.layers.get_all_params(l_critic1) + lasagne.layers.get_all_params(l_critic2)
    params = params_actor + params_crit
    # get target params
    params_target = lasagne.layers.get_all_params(l_actor_target1) + \
                    lasagne.layers.get_all_params(l_actor_target2) + \
                    lasagne.layers.get_all_params(l_critic_target1) + \
                    lasagne.layers.get_all_params(l_critic_target2)

    # set critic target to critic params
    for param, param_target in zip(params, params_target):
        param_target.set_value(param.get_value())

    # calculate grads and steps
    grads_actor = T.grad(actor_loss, params_actor)
    grads_critic = T.grad(critic_loss, params_crit)
    grads_actor = lasagne.updates.total_norm_constraint(grads_actor, 10)
    grads_critic = lasagne.updates.total_norm_constraint(grads_critic, 10)

    actor_lr = theano.shared(lasagne.utils.floatX(actor_lr))
    critic_lr = theano.shared(lasagne.utils.floatX(critic_lr))
    actor_updates = lasagne.updates.adam(grads_actor, params_actor, actor_lr, 0.9, 0.99)
    critic_updates = lasagne.updates.adam(grads_critic, params_crit, critic_lr, 0.9, 0.99)
    updates = OrderedDict()
    updates.update(actor_updates)
    updates.update(critic_updates)

    # target function update
    target_updates = OrderedDict()
    for param, param_target in zip(params, params_target):
        update = (1. - target_update_coeff) * param_target + target_update_coeff * param
        target_updates[param_target] = update

    # compile theano functions
    train_fn = theano.function([states, actions, rewards, terminals, next_states],
                               [actor_loss, critic_loss], updates=updates)
    actor_fn = theano.function([states], actions_pred)
    target_update_fn = theano.function([], updates=target_updates)

    return train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr


class Agent(object):
    def __init__(self, actor_fn, params_actor, params_crit):
        self._actor_fn = actor_fn
        self.params_actor = params_actor
        self.params_crit = params_crit

    def get_actor_weights(self):
        return [p.get_value() for p in self.params_actor]

    def get_critic_weights(self):
        return [p.get_value() for p in self.params_crit]

    def get_weights(self):
        actor_weights = self.get_actor_weights()
        crit_weights = self.get_critic_weights()
        return actor_weights, crit_weights

    def set_actor_weights(self, weights):
        assert len(weights) == len(self.params_actor)
        [p.set_value(w) for p, w in zip(self.params_actor, weights)]

    def set_crit_weights(self, weights):
        assert len(weights) == len(self.params_crit)
        [p.set_value(w) for p, w in zip(self.params_crit, weights)]

    def set_weights(self, actor_weights, crit_weights):
        self.set_actor_weights(actor_weights)
        self.set_crit_weights(crit_weights)

    def save(self, fname):
        with open(fname, 'wb') as f:
            actor_weigths = self.get_actor_weights()
            crit_weigths = self.get_critic_weights()
            cPickle.dump([actor_weigths, crit_weigths], f, -1)

    def load(self, fname):
        with open(fname, 'rb') as f:
            actor_weights, critic_wieghts = cPickle.load(f)
            self.set_actor_weights(actor_weights)
            self.set_crit_weights(critic_wieghts)

    def act(self, state):
        state = np.asarray([state])
        return self._actor_fn(state)[0]
