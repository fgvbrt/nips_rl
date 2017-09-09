import cPickle
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
from collections import OrderedDict
from lasagne.layers import batch_norm
from lasagne.layers import DenseLayer
from lasagne.layers import get_all_param_values
from lasagne.layers import set_all_param_values

floatX = theano.config.floatX


class DeepLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, memory, batch_size, discount):

        self.memory = memory

        self.batch_size = batch_size

        self.state_shape = (batch_size, self.memory.state_size)
        self.actions_shape = (batch_size, self.memory.action_size)
        self.rewards_shape = (batch_size, 1)
        self.terminals_shape = (batch_size, 1)

        self.discount = discount
        self.tau = 0.001
        self.lr_actor = 0.00025
        self.lr_critic = self.lr_actor
        self.momentum = 0
        self.clip_delta = 1.0

        lasagne.random.set_rng(self.memory.rng)

        self.update_counter = 0
        self.debug_frequency = 10000

        self.l_full, self.l_critic, self.l_actor, l_s_in, l_a_in = self.build_network()
        self.l_target, _, _, l_s_target_in, l_a_target_in = self.build_network()

        set_all_param_values(self.l_target, get_all_param_values(self.l_full))

        states = T.matrix('states')
        actions = T.matrix('actions')
        rewards = T.col('rewards')
        next_states = T.matrix('next_states')
        terminals = T.col('terminals')

        self._states = theano.shared(np.zeros(self.state_shape, dtype=floatX))
        self._actions = theano.shared(np.zeros(self.actions_shape, dtype=floatX))
        self._rewards = theano.shared(np.zeros(self.rewards_shape, dtype=floatX),
                                      broadcastable=(False, True))
        self._next_states = theano.shared(np.zeros(self.state_shape, dtype=floatX))
        self._terminals = theano.shared(np.zeros(self.terminals_shape, dtype=floatX),
                                        broadcastable=(False, True))

        # Critic
        v_vals = lasagne.layers.get_output(
            self.l_critic,
            {l_s_in: states, l_a_in: actions},
            deterministic=False)

        t_vals = lasagne.layers.get_output(
            self.l_target,
            {l_s_target_in: next_states},
            deterministic=False)

        td = rewards + (T.ones_like(terminals) - terminals) * self.discount * t_vals - v_vals

        if self.clip_delta > 0:
            # If we simply take the squared clipped td as our loss,
            # then the gradient will be zero whenever the td exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            #
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped td, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(td), self.clip_delta)
            linear_part = abs(td) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * td ** 2

        loss = T.mean(loss)

        params_critic = lasagne.layers.helper.get_all_params(self.l_critic, trainable=True)
        updates_critic = lasagne.updates.adam(loss, params_critic, self.lr_critic)
        if self.momentum > 0:
            updates_critic = lasagne.updates.apply_momentum(updates_critic,
                                                            None, self.momentum)
        givens = {
            states: self._states,
            actions: self._actions,
            rewards: self._rewards,
            next_states: self._next_states,
            terminals: self._terminals
        }
        self._train_critic = theano.function([], loss, updates=updates_critic,
                                             givens=givens)

        grads_critic = T.grad(loss, params_critic)
        self._grads_critic = theano.function([], grads_critic, givens=givens)

        # Actor
        q_vals = lasagne.layers.get_output(
            self.l_full,
            {l_s_in: states},
            deterministic=False)

        loss_actor = -T.mean(q_vals)

        params_actor = lasagne.layers.helper.get_all_params(self.l_actor, trainable=True)
        updates_actor = lasagne.updates.adam(loss_actor, params_actor, self.lr_actor)
        if self.momentum > 0:
            updates_actor = lasagne.updates.apply_momentum(updates_actor,
                                                           None, self.momentum)
        givens = {
            states: self._states,
        }
        self._train_actor = theano.function([], loss_actor, updates=updates_actor,
                                            givens=givens)

        a_vals = lasagne.layers.get_output(
            self.l_actor,
            {l_s_in: states},
            deterministic=True)

        self._a_vals = theano.function([], a_vals, givens=givens)
        grads_actor = T.grad(loss_actor, params_actor)
        self._grads_actor = theano.function([], grads_actor, givens=givens)

        # Helper
        updates_target = OrderedDict()
        full_params = lasagne.layers.get_all_params(self.l_full)
        curr_params = lasagne.layers.get_all_params(self.l_target)
        for curr_param, full_param in zip(curr_params, full_params):
            updates_target[curr_param] = self.tau * full_param + (1 - self.tau) * curr_param
        self._update_target = theano.function([], [], updates=updates_target)
        self.action_counter = 0

    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.
        """

        self._states.set_value(states)
        self._actions.set_value(actions)
        self._rewards.set_value(rewards)

        self._next_states.set_value(next_states)
        self._terminals.set_value(terminals)

        loss_critic = self._train_critic()
        if np.isnan(loss_critic):
            raise TypeError("Critic's loss function return NaN!")
        loss_actor = self._train_actor()
        if np.isnan(loss_actor):
            raise TypeError("Actor's loss function return NaN!")

        self._update_target()
        self.update_counter += 1

        if self.update_counter % self.debug_frequency == 0:
            self.debug_network()
            print "Step {:d}, td-error {:.5f}, q-value {:.5f}".format(
                self.update_counter, np.sqrt(loss_critic), -loss_actor
            )

    def choose_action(self, state):
        _state = np.zeros(self.state_shape, dtype=floatX)
        _state[0, ...] = state
        self._states.set_value(_state)
        a = self._a_vals()[0]
        return a

    def build_network(self):
        """
        Build a network based on actor-critic scheme.
        """

        l_state_in = lasagne.layers.InputLayer(self.state_shape)
        l_action_in = lasagne.layers.InputLayer(self.actions_shape)

        # Critic
        l_critic0 = lasagne.layers.ConcatLayer([l_state_in, l_action_in], 1)
        l_critic1 = DenseLayer(
            l_critic0,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        l_critic2 = DenseLayer(
            l_critic1,
            num_units=20,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        l_critic3 = DenseLayer(
            l_critic2,
            num_units=1,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        # Actor
        l_actor1 = DenseLayer(
            l_state_in,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        l_actor2 = DenseLayer(
            l_actor1,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        l_actor3 = DenseLayer(
            l_actor2,
            num_units=self.memory.action_size,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        # Join
        l_join = lasagne.layers.ConcatLayer([l_state_in, l_actor3], 1)
        l_join = DenseLayer(
            l_join,
            num_units=l_critic1.num_units,
            nonlinearity=l_critic1.nonlinearity,
            W=l_critic1.W, b=l_critic1.b
        )
        l_join = DenseLayer(
            l_join,
            num_units=l_critic2.num_units,
            nonlinearity=l_critic2.nonlinearity,
            W=l_critic2.W, b=l_critic2.b
        )
        l_join = DenseLayer(
            l_join,
            num_units=l_critic3.num_units,
            nonlinearity=l_critic3.nonlinearity,
            W=l_critic3.W, b=l_critic3.b
        )

        return l_join, l_critic3, l_actor3, l_state_in, l_action_in

    def debug_param_values(self, l_name, params):
        w1 = np.abs(params[0])
        w2 = np.abs(params[len(params)/2])
        w3 = np.abs(params[-2])
        print "{}\t[{:.4f}, {:.4f}] [{:.4f}, {:.4f}] [{:.4f}, {:.4f}]".format(
            l_name,
            np.mean(w1), np.max(w1),
            np.mean(w2), np.max(w2),
            np.mean(w3), np.max(w3)
        )

    def debug_network(self):
        params = get_all_param_values(self.l_critic, trainable=True)
        print "Three layers:\t{} {} {}".format(
            params[0].shape, params[len(params)/2].shape, params[-2].shape
        )
        self.debug_param_values('Critic  W:', params)
        params = self._grads_critic()
        self.debug_param_values('Critic dW:', params)
        params = get_all_param_values(self.l_actor, trainable=True)
        self.debug_param_values('Actor  W:', params)
        params = self._grads_actor()
        self.debug_param_values('Actor dW:', params)

    def save_model(self, filename):
        with open(filename, 'w') as file:
            params = {
                'params_full': get_all_param_values(self.l_full),
                'params_tar': get_all_param_values(self.l_target),
            }
            cPickle.dump(params, file)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            params = cPickle.load(file)
            set_all_param_values(self.l_full,  params['params_full'])
            set_all_param_values(self.l_target, params['params_tar'])
