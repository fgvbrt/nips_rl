import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict
import numpy as np


def build_actor(state_size,  num_act,
                last_nonlinearity1=lasagne.nonlinearities.sigmoid,
                last_nonlinearity2=lasagne.nonlinearities.softplus,
                hid_sizes=(300, 200, 100),
                nonlinearity=lasagne.nonlinearities.tanh):
    l_input = lasagne.layers.InputLayer((None, state_size))
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(
            l_hid, hid_size,
            nonlinearity=nonlinearity
        )

    l_last = lasagne.layers.DenseLayer(l_hid, num_act * 2, nonlinearity=None)

    l_mu = lasagne.layers.SliceLayer(l_last, indices=slice(0, num_act))
    #l_mu = lasagne.layers.NonlinearityLayer(l_mu, last_nonlinearity1)

    l_sigma = lasagne.layers.SliceLayer(l_last, indices=slice(num_act, None))
    l_sigma = lasagne.layers.NonlinearityLayer(l_sigma, last_nonlinearity2)

    return l_input, l_mu, l_sigma


def build_critic(state_size, hid_sizes=(100, 50, 20),
                 nonlinearity=lasagne.nonlinearities.tanh):
    l_input = lasagne.layers.InputLayer((None, state_size))
    l_hid = l_input
    for hid_size in hid_sizes:
        l_hid = lasagne.layers.DenseLayer(
            l_hid, hid_size,
            nonlinearity=nonlinearity
        )
    l_baseline = lasagne.layers.DenseLayer(l_hid, 1, nonlinearity=None)
    return l_input, l_baseline


def build_model(state_size, num_act,
                alpha=1e-3,
                actor_lr=0.00025,
                critic_lr=0.0005,
                entropy_coeff=0.0,
                ):

    # input tensors
    states = T.matrix('states')
    actions = T.matrix('actions')
    targets = T.vector('targets')

    # create running mean and std
    state_mean = theano.shared(np.zeros(state_size, dtype='float32'))
    state_var = theano.shared(np.ones(state_size, dtype='float32'))
    state_norm = (states - state_mean) / (T.sqrt(state_var) + 1e-5)

    # build actor and critic networks
    l_actor_in, l_mu, l_sigma = build_actor(state_size, num_act)
    l_critic_in, l_critic = build_critic(state_size)

    # get output tensors
    mu = lasagne.layers.get_output(l_mu, state_norm)
    sigma = lasagne.layers.get_output(l_sigma, state_norm)
    baseline = lasagne.layers.get_output(l_critic, state_norm)

    # critic/baseline loss
    td_error = targets - T.flatten(baseline)
    critic_loss = 0.5 * (td_error ** 2)
    critic_loss = T.mean(critic_loss)

    # policy loss
    entropy = 0.5*num_act*(1. + T.log(2.*np.pi)) + T.sum(T.log(sigma + 1e-5), axis=1)
    log_prob = -1. * T.sum(T.log(sigma + 1e-5), axis=1) \
               - 0.5*(num_act*T.log(2.*np.pi) + T.sum(((actions-mu)/sigma)**2, axis=1))
    adv = theano.gradient.disconnected_grad(td_error)
    #adv = targets
    # normilize advantage
    adv = (adv - T.mean(adv)) / (T.std(adv) + 1e-5)
    actor_loss = -1. * (log_prob * adv + entropy_coeff*entropy)
    actor_loss = T.mean(actor_loss)

    params_actor = lasagne.layers.get_all_params(l_mu)
    params_crit = lasagne.layers.get_all_params(l_critic)

    # calculate grads and steps
    grads_actor = T.grad(actor_loss, params_actor)
    grads_critic = T.grad(critic_loss, params_crit)
    grads_actor = lasagne.updates.total_norm_constraint(grads_actor, 10)
    grads_critic = lasagne.updates.total_norm_constraint(grads_critic, 10)

    actor_updates = lasagne.updates.adam(grads_actor, params_actor, actor_lr)
    critic_updates = lasagne.updates.adam(grads_critic, params_crit, critic_lr)
    updates = OrderedDict()
    updates.update(actor_updates)
    updates.update(critic_updates)

    #diff = T.flatten(states) - state_mean
    #incr = alpha*diff
    #updates[state_mean] = state_mean + incr
    #updates[state_var] = (1.-alpha)*(state_var + diff*incr)
    updates[state_mean] = (1.-alpha)*state_mean + alpha*T.mean(states, axis=0)
    updates[state_var] = (1.-alpha) * state_var + alpha*T.var(states, axis=0)

    train_fn = theano.function([states, actions, targets], actor_loss, updates=updates)
    actor_fn = theano.function([states], [mu, sigma])
    value_fn = theano.function([states], baseline)

    return train_fn, actor_fn, value_fn, params_actor + [state_mean, state_var]
