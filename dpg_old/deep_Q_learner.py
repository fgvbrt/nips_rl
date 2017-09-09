import os
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


class DeepQLearner():
    def __init__(self, l_sizes, enemy_shape, units_shape, masks_shape, screen_shape, action_size,
                 n_heads, rng, batch_accumulator='mean'):

        self.batch_size = 32
        self.enemy_shape = tuple([self.batch_size] + enemy_shape)
        self.units_shape = tuple([self.batch_size] + units_shape)
        self.masks_shape = tuple([self.batch_size] + masks_shape)

        self.screens_shape = tuple([self.batch_size] + screen_shape)

        self.actions_shape = (self.batch_size, 1)
        self.rewards_shape = (self.batch_size, 1)
        self.terminals_shape = (self.batch_size, 1)

        self.action_size = action_size
        self.n_heads = n_heads
        self.double_dqn = True
        self.rng = rng
        self.batch_size = 32
        self.discount = 0.99
        self.tau = 0.001
        self.update_interval = 1
        self.lr = 0.0001
        self.momentum = 0
        self.clip_delta = 1.0

        lasagne.random.set_rng(self.rng)
        self.update_counter = 0
        self.debug_frequency = 1000

        l_sizes.append(action_size)
        self.l_out, l_in1, l_in2, l_in3, l_in4, l_in5 = self.build_network(l_sizes)

        # create target network
        self.l_target, l2_in1, l2_in2, l2_in3, l2_in4, l2_in5 = self.build_network(l_sizes)
        self._update_target(1)

        enemy = T.tensor3('enemy')
        enemy_mask = T.matrix('enemy_mask')
        units = T.tensor3('units')
        units_masks = T.matrix('units_masks')
        screens = T.tensor4('screens')
        actions = T.icol('actions')
        rewards = T.col('rewards')
        next_enemy = T.tensor3('next_enemy')
        next_enemy_mask = T.matrix('next_enemy_mask')
        next_units = T.tensor3('next_units')
        next_units_mask = T.matrix('next_units_mask')
        next_screens = T.tensor4('next_screens')
        terminals = T.icol('terminals')

        self._enemy = theano.shared(np.zeros(self.enemy_shape, dtype=floatX))
        self._enemy_mask = theano.shared(np.zeros(self.masks_shape, dtype=floatX))
        self._units = theano.shared(np.zeros(self.units_shape, dtype=floatX))
        self._units_masks = theano.shared(np.zeros(self.masks_shape, dtype=floatX))
        self._screens = theano.shared(np.zeros(self.screens_shape, dtype=floatX))

        self._actions = theano.shared(np.zeros(self.actions_shape, dtype='int32'),
                                      broadcastable=(False, True))
        self._rewards = theano.shared(np.zeros(self.rewards_shape, dtype=floatX),
                                      broadcastable=(False, True))

        self._next_enemy = theano.shared(np.zeros(self.enemy_shape, dtype=floatX))
        self._next_enemy_mask = theano.shared(np.zeros(self.masks_shape, dtype=floatX))
        self._next_units = theano.shared(np.zeros(self.units_shape, dtype=floatX))
        self._next_units_masks = theano.shared(np.zeros(self.masks_shape, dtype=floatX))
        self._next_screens = theano.shared(np.zeros(self.screens_shape, dtype=floatX))

        self._terminals = theano.shared(np.zeros(self.terminals_shape, dtype='int32'),
                                        broadcastable=(False, True))

        q_vals = lasagne.layers.get_output(self.l_out, {l_in1: enemy, l_in2: enemy_mask,
                                                        l_in3: units, l_in4: units_masks,
                                                        l_in5: screens})

        next_q_vals = lasagne.layers.get_output(self.l_target, {l2_in1: next_enemy, l2_in2: next_enemy_mask,
                                                                l2_in3: next_units, l2_in4: next_units_mask,
                                                                l2_in5: next_screens})
        if self.double_dqn > 0:
            next_q_vals_c = lasagne.layers.get_output(self.l_out, {l_in1: next_enemy, l_in2: next_enemy_mask,
                                                                   l_in3: next_units, l_in4: next_units_mask,
                                                                   l_in5: next_screens})
            actions_next = T.argmax(next_q_vals_c.reshape(
                (self.n_heads * self.batch_size, self.action_size)), axis=1)

            target = rewards + (T.ones_like(terminals) - terminals) * self.discount * \
                               next_q_vals.reshape((self.n_heads * self.batch_size, self.action_size)) \
                                   [T.arange(self.batch_size * self.n_heads), actions_next].reshape((-1, self.n_heads))

        else:
            target = rewards + (T.ones_like(terminals) - terminals) * self.discount * \
                               T.max(next_q_vals.reshape((self.n_heads * self.batch_size, self.action_size)),
                                     axis=1, keepdims=True).reshape((-1, self.n_heads))

        # take into account number of heads
        diff = target - q_vals.reshape((self.n_heads * self.batch_size, self.action_size)) \
            [T.arange(self.batch_size * self.n_heads),
             actions.reshape((-1,)).repeat(self.n_heads)].reshape((-1, self.n_heads))

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            #
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(loss, params, self.lr)
        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates,
                                                     None, self.momentum)
        givens = {
            enemy: self._enemy,
            enemy_mask: self._enemy_mask,
            units: self._units,
            units_masks: self._units_masks,
            screens: self._screens,
            actions: self._actions,
            rewards: self._rewards,
            next_enemy: self._next_enemy,
            next_enemy_mask: self._next_enemy_mask,
            next_units: self._next_units,
            next_units_mask: self._next_units_masks,
            next_screens: self._next_screens,
            terminals: self._terminals
        }

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)

        grads = T.grad(loss, params)
        self._grads = theano.function([], grads, givens=givens)

        self._q_vals = theano.function([], q_vals, givens={enemy: self._enemy,
                                                           enemy_mask: self._enemy_mask,
                                                           units: self._units,
                                                           units_masks: self._units_masks,
                                                           screens: self._screens})

    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.
        Arguments:
        states - batch_size x state_size
        screens - batch_size x screen_size
        actions - batch_size x action_size
        rewards - batch_size x 1
        next_states - batch_size x state_size
        next_screens - batch_size x screen_size
        """

        self._enemy.set_value(states[0])
        self._enemy_mask.set_value(states[1])
        self._units.set_value(states[2])
        self._units_masks.set_value(states[3])
        self._screens.set_value(states[4])

        self._actions.set_value(actions)
        self._rewards.set_value(rewards)

        self._next_enemy.set_value(next_states[0])
        self._next_enemy_mask.set_value(next_states[1])
        self._next_units.set_value(next_states[2])
        self._next_units_masks.set_value(next_states[3])
        self._next_screens.set_value(next_states[4])

        self._terminals.set_value(terminals)

        loss, q_val = self._train()
        if np.isnan(loss):
            raise TypeError("Loss function return NaN!")

        if self.update_interval > 0 and self.update_counter % self.update_interval == 0:
            self._update_target(self.tau)

        if self.update_counter % self.debug_frequency == 0:
            self.debug_network()
            print "Step {:d}, td-error {:.5f}, q-value {:.5f} ".format(
                self.update_counter, np.sqrt(loss), np.mean(q_val)
            )

        self.update_counter += 1

        return np.sqrt(loss), q_val

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(self.action_size)

        enemy = np.zeros(self.enemy_shape, dtype=floatX)
        enemy_mask = np.zeros(self.masks_shape, dtype=floatX)
        units = np.zeros(self.units_shape, dtype=floatX)
        units_masks = np.zeros(self.masks_shape, dtype=floatX)
        screens = np.zeros(self.screens_shape, dtype=floatX)

        enemy[0, ...] = state[0]
        enemy_mask[0, ...] = state[1]
        units[0, ...] = state[2]
        units_masks[0, ...] = state[3]
        screens[0, ...] = state[4]

        self._enemy.set_value(enemy)
        self._enemy_mask.set_value(enemy_mask)
        self._units.set_value(units)
        self._units_masks.set_value(units_masks)
        self._screens.set_value(screens)

        return self._q_vals()[0].reshape((self.n_heads, -1)).argmax(axis=1)

    def _update_target(self, tau):
        weights_cur = lasagne.layers.helper.get_all_param_values(self.l_out)
        weights_tar = lasagne.layers.helper.get_all_param_values(self.l_target)

        weights = []
        for i in range(len(weights_cur)):
            weights.append(tau * weights_cur[i] + (1 - tau) * weights_tar[i])
        lasagne.layers.helper.set_all_param_values(self.l_target, weights)

    def build_network_screen(self, l_screen):
        l_input = lasagne.layers.ReshapeLayer(l_screen,
                                              (self.batch_size, 1, 84, 84)
                                              )

        # if theano.sandbox.cuda.cuda_enabled:
        if theano.sandbox.cuda.dnn.dnn_available():
            from lasagne.layers import dnn
            conv_layer = dnn.Conv2DDNNLayer
        else:
            conv_layer = lasagne.layers.Conv2DLayer

        l_conv1 = batch_norm(conv_layer(
            l_input,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        ))
        l_conv2 = batch_norm(conv_layer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        ))
        l_dense1 = batch_norm(DenseLayer(
            l_conv2,
            num_units=96,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        ))

        return l_dense1

    def build_network_units(self, l_units, l_masks):
        l_enc = lasagne.layers.GRULayer(l_units, mask_input=l_masks,
                                        num_units=64, name='GRUEncoder'
                                        )
        l_hid = lasagne.layers.SliceLayer(l_enc, indices=-1, axis=1)

        l_dense = batch_norm(DenseLayer(
            l_hid,
            num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        ))
        return l_dense

    def build_network_state(self, l_enemy, l_enemy_mask, l_units, l_units_masks, l_screen):
        """
        Build state pathway from enemy+units+screen.
        """
        l_enemy = self.build_network_units(l_enemy, l_enemy_mask)
        l_units = self.build_network_units(l_units, l_units_masks)
        l_screen = self.build_network_screen(l_screen)
        l_state = lasagne.layers.ConcatLayer([l_enemy, l_units, l_screen], 1)
        l_state = batch_norm(DenseLayer(
            l_state,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        ))
        return l_state

    def build_network(self, l_sizes):
        l_in1 = lasagne.layers.InputLayer(self.enemy_shape)
        l_in2 = lasagne.layers.InputLayer(self.enemy_shape)
        l_in3 = lasagne.layers.InputLayer(self.units_shape)
        l_in4 = lasagne.layers.InputLayer(self.masks_shape)
        l_in5 = lasagne.layers.InputLayer(self.screens_shape)

        l_in = self.build_network_state(l_in1, l_in2, l_in3, l_in4, l_in5)

        layers = []
        for k, p in enumerate(l_sizes):
            lk = []
            if k == 0:
                for i in range(self.n_heads):
                    lk.append(DenseLayer(
                        l_in,
                        num_units=p,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeUniform(),
                        b=lasagne.init.Constant(.1)
                    ))
            elif k == len(l_sizes) - 1:
                for i in range(self.n_heads):
                    lk.append(DenseLayer(
                        layers[k - 1][i],
                        num_units=p,
                        nonlinearity=None,
                        W=lasagne.init.HeUniform(),
                        b=lasagne.init.Constant(.1)
                    ))
            else:
                for i in range(self.n_heads):
                    lk.append(DenseLayer(
                        layers[k - 1][i],
                        num_units=p,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeUniform(),
                        b=lasagne.init.Constant(.1)
                    ))
            layers.append(lk)
        # concat all final layers
        network = lasagne.layers.ConcatLayer(layers[-1], 1)

        return network, l_in1, l_in2, l_in3, l_in4, l_in5

    def debug_param_values(self, l_name, params):
        w1 = np.abs(params[0])
        w2 = np.abs(params[3])
        print "{}\t{}=[{:.6f}, {:.6f}], {}=[{:.6f}, {:.6f}]".format(
            l_name,
            w1.shape, np.mean(w1), np.max(w1),
            w2.shape, np.mean(w2), np.max(w2)
        )

    def debug_network(self):
        params = get_all_param_values(self.l_out, trainable=True)
        self.debug_param_values('W:', params)
        params = self._grads()
        self.debug_param_values('dW:', params)

    def save_model(self, filename):
        with open(filename, 'w') as file:
            params = {
                'params_out': get_all_param_values(self.l_out),
                'params_tar': get_all_param_values(self.l_target),
            }
            cPickle.dump(params, file)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            params = cPickle.load(file)
            set_all_param_values(self.l_out, params['params_out'])
            set_all_param_values(self.l_target, params['params_tar'])


def main():
    BATCH = 32
    NUMBER_FEATURES_UNIT = 20
    NUMBER_FEATURES_ENEMY = 20 + 8
    NUMBER_UNITS = 6
    SCREEN_CHANNEL = 1
    SCREEN_HEIGHT = 84
    SCREEN_WIDTH = 84
    NUMBER_ACTIONS = 6
    NUMBER_HEADS = 3

    layers = [256, 20]

    # prev action True
    rng = np.random.RandomState()
    net = DeepQLearner(layers, [NUMBER_UNITS, NUMBER_FEATURES_ENEMY],
                       [NUMBER_UNITS, NUMBER_FEATURES_UNIT],
                       [NUMBER_UNITS],
                       [SCREEN_CHANNEL, SCREEN_HEIGHT, SCREEN_WIDTH],
                       NUMBER_ACTIONS, NUMBER_HEADS, rng)

    params = lasagne.layers.helper.get_all_params(net.l_out)
    print "Parameters"
    print "-" * 40
    for param in params:
        print param, param.get_value().shape
    print "-" * 40

    enemy = np.random.randn(BATCH, NUMBER_UNITS, NUMBER_FEATURES_ENEMY).astype(floatX)
    enemy_mask = np.zeros((BATCH, NUMBER_UNITS)).astype(floatX)
    units = np.random.randn(BATCH, NUMBER_UNITS, NUMBER_FEATURES_UNIT).astype(floatX)
    masks = np.zeros((BATCH, NUMBER_UNITS)).astype(floatX)
    screens = np.random.randn(BATCH, SCREEN_CHANNEL, SCREEN_HEIGHT, SCREEN_WIDTH).astype(floatX)

    actions = np.random.randint(0, NUMBER_ACTIONS, BATCH).reshape((BATCH, 1)).astype('int32')
    rewards = np.random.randn(BATCH, 1).astype(floatX)

    next_enemy = np.random.randn(BATCH, NUMBER_UNITS, NUMBER_FEATURES_ENEMY).astype(floatX)
    next_enemy_mask = np.zeros((BATCH, NUMBER_UNITS)).astype(floatX)
    next_units = np.random.randn(BATCH, NUMBER_UNITS, NUMBER_FEATURES_UNIT).astype(floatX)
    next_masks = np.zeros((BATCH, NUMBER_UNITS)).astype(floatX)
    next_screens = np.random.randn(BATCH, SCREEN_CHANNEL, SCREEN_HEIGHT, SCREEN_WIDTH).astype(floatX)

    terminals = np.random.randint(0, 2, BATCH).reshape((BATCH, 1)).astype('int32')

    # create mask
    for i in range(BATCH):
        enemy_mask[i, :np.random.randint(0, NUMBER_UNITS)] = 1
        masks[i, :np.random.randint(0, NUMBER_UNITS)] = 1
        next_enemy_mask[i, :np.random.randint(0, NUMBER_UNITS)] = 1
        next_masks[i, :np.random.randint(0, NUMBER_UNITS)] = 1

    states = [enemy, enemy_mask, units, masks, screens]
    next_states = [next_enemy, next_enemy_mask, next_units, next_masks, next_screens]
    state = [enemy[0], enemy_mask[0], units[0], masks[0], screens[0]]

    # check actions
    action = net.choose_action(state, 0.0)
    print action, np.random.choice(np.arange(action.max() + 1)[np.bincount(action) == np.bincount(action).max()])
    action = net.choose_action(state, 1.0)
    print action

    # check loss
    tr_start = time.time()
    loss = net.train(states, actions, rewards, next_states, terminals)
    tr_end = time.time()
    print loss, tr_end - tr_start

    net.debug_network()


if __name__ == '__main__':
    main()
