import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict
import cPickle
import numpy as np
from lasagne.layers import Layer, DenseLayer, NonlinearityLayer
from lasagne import init


class LayerNorm(Layer):
    def __init__(self, incoming, epsilon=1e-4, beta=init.Constant(0), gamma=init.Constant(1), **kwargs):
        super(LayerNorm, self).__init__(incoming, **kwargs)

        self.epsilon = epsilon
        
        n_features = self.input_shape[1]
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, (n_features,), 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, (n_features,), 'gamma',
                                        trainable=True, regularizable=True)

    def get_output_for(self, input, **kwargs):
        input_mean = T.mean(input, axis=1, keepdims=True)
        input_inv_std = T.inv(T.sqrt(T.var(input, axis=1, keepdims=True) + self.epsilon))
        return (input - input_mean) * input_inv_std * self.gamma + self.beta


def build_actor(l_input, num_act, last_nonlinearity=lasagne.nonlinearities.sigmoid,
                hid_sizes=(64, 64), layer_norm=True,
                nonlinearity=lasagne.nonlinearities.elu):
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(l_hid, hid_size)
        if layer_norm:
            l_hid = LayerNorm(l_hid)
        l_hid = NonlinearityLayer(l_hid, nonlinearity)

    return lasagne.layers.DenseLayer(l_hid, num_act, nonlinearity=last_nonlinearity)


def build_critic(l_input, hid_sizes=(64, 32), layer_norm=True,
                 nonlinearity=lasagne.nonlinearities.elu):
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(l_hid, hid_size)
        if layer_norm:
            l_hid = LayerNorm(l_hid)
        l_hid = NonlinearityLayer(l_hid, nonlinearity)

    return lasagne.layers.DenseLayer(l_hid, 1, nonlinearity=None)


def build_actor_critic(state_size, num_act, layer_norm):
    # input layers
    l_states = lasagne.layers.InputLayer([None, state_size])
    l_actions = lasagne.layers.InputLayer([None, num_act])
    l_input_critic = lasagne.layers.ConcatLayer([l_states, l_actions])
    # actor layer
    l_actor = build_actor(l_states, num_act, layer_norm=layer_norm)
    # critic layer
    l_critic = build_critic(l_input_critic, layer_norm=layer_norm)
    return l_states, l_actions, l_actor, l_critic


def build_model(state_size, num_act, gamma=0.99,
                actor_lr=0.00025,
                critic_lr=0.0005,
                target_update_coeff=1e-4,
                clip_delta=10.,
                layer_norm=True):

    # input tensors
    states = T.matrix('states')
    next_states = T.matrix('next_states')
    actions = T.matrix('actions')
    rewards = T.col('rewards')
    terminals = T.col('terminals')

    # current network
    l_states, l_actions, l_actor, l_critic = build_actor_critic(state_size, num_act, layer_norm)
    # target network
    l_states_target, l_actions_target, l_actor_target, l_critic_target =\
        build_actor_critic(state_size, num_act, layer_norm)

    # get current network output tensors
    actions_pred = lasagne.layers.get_output(l_actor, states)
    q_vals = lasagne.layers.get_output(l_critic, {l_states: states, l_actions: actions})
    v_vals = lasagne.layers.get_output(l_critic, {l_states: states, l_actions: actions_pred})

    # get target network q-values
    actions_pred_target = lasagne.layers.get_output(l_actor_target, next_states)
    v_vals_target = lasagne.layers.get_output(
        l_critic_target,
        {l_states_target: next_states, l_actions_target: actions_pred_target})

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
    params_actor = lasagne.layers.get_all_params(l_actor)
    params_crit = lasagne.layers.get_all_params(l_critic)
    params = params_actor + params_crit
    # get target params
    params_target = lasagne.layers.get_all_params(l_actor_target) + \
                    lasagne.layers.get_all_params(l_critic_target)

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
        self.params_actor_no_norm =  [p for p in params_actor if p.name not in ('gamma', 'beta')]
        self.params_crit = params_crit

    def get_actor_weights(self, exclude_norm=False):
        if exclude_norm:
            params = self.params_actor_no_norm
        else:
            params = self.params_actor
        return [p.get_value() for p in params]

    def get_critic_weights(self):
        return [p.get_value() for p in self.params_crit]

    def get_weights(self):
        actor_weights = self.get_actor_weights()
        crit_weights = self.get_critic_weights()
        return actor_weights, crit_weights

    def set_actor_weights(self, weights, exclude_norm=False):
        if exclude_norm:
            params = self.params_actor_no_norm
        else:
            params = self.params_actor
        assert len(weights) == len(params)
        [p.set_value(w) for p, w in zip(params, weights)]

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

    def act_batch(self, states):
        return self._actor_fn(states)
