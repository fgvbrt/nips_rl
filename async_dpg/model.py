import numpy as np
import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict


def get_adam_steps_and_updates(all_grads, params, learning_rate=0.001,
                               beta1=0.9, beta2=0.99, epsilon=1e-8):
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    adam_steps = []
    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t

        adam_steps.append(step)

    updates[t_prev] = t
    return adam_steps, updates


def build_actor(l_input, num_act, last_nonlinearity=lasagne.nonlinearities.sigmoid,
                hid_sizes=(300, 100, 10),
                nonlinearity=lasagne.nonlinearities.elu):
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(
            l_hid, hid_size,
            nonlinearity=nonlinearity
        )

    return lasagne.layers.DenseLayer(l_hid, num_act, nonlinearity=last_nonlinearity)


def build_critic(l_input, hid_sizes=(100, 50),
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


def build_model(state_size, num_act, learning_rate=0.00025,
                clip_delta=1, target_update_coeff=1e-3):

    # input tensors
    states = T.matrix('states')
    actions = T.matrix('actions')
    q_targets = T.vector('q_targets')

    # current network
    l_states, l_actions, l_actor, l_critic = build_actor_critic(state_size, num_act)
    # target network
    l_states_target, l_actions_target, l_actor_target, l_critic_target = build_actor_critic(state_size, num_act)

    # get current network output tensors
    actions_pred = lasagne.layers.get_output(l_actor, states)
    q_vals = T.flatten(lasagne.layers.get_output(l_critic, {l_states: states, l_actions: actions}))
    v_vals = T.flatten(lasagne.layers.get_output(l_critic, {l_states: states, l_actions: actions_pred}))

    # get target network q-values
    actions_pred_target = lasagne.layers.get_output(l_actor_target, states)
    v_vals_target = T.flatten(lasagne.layers.get_output(
        l_critic_target,
        {l_states_target: states, l_actions_target: actions_pred_target})
    )

    # critic loss
    td_error = q_targets - q_vals
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
    params_target = lasagne.layers.get_all_params(l_actor_target) +\
        lasagne.layers.get_all_params(l_critic_target)

    # set critic target to critic params
    for param, param_target in zip(params, params_target):
        param_target.set_value(param.get_value())

    # calculate grads and steps
    grads_actor = T.grad(actor_loss, params_actor)
    grads_critic = T.grad(critic_loss, params_crit)
    grads = grads_actor + grads_critic
    grads = lasagne.updates.total_norm_constraint(grads, 10)
    steps, updates = get_adam_steps_and_updates(grads, params, learning_rate)
    steps_fn = theano.function([states, actions, q_targets], steps, updates=updates)

    # actions and value function
    actor_fn = theano.function([states], actions_pred)
    val_fn = theano.function([states], v_vals_target)

    # target function update
    target_updates = OrderedDict()
    for param, param_target in zip(params, params_target):
        update = (1. - target_update_coeff)*param_target + target_update_coeff*param
        target_updates[param_target] = update
    target_update_fn = theano.function([], updates=target_updates)

    return steps_fn, actor_fn, val_fn, target_update_fn, params, params_target
