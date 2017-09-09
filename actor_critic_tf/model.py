import tensorflow as tf
import numpy as np
import cPickle
from utils import softplus, gaussian, sigmoid
from distributions import DiagGaussianPd, BernoulliPd
from scipy.stats import bernoulli


def build_fc_net(inp, hid_sizes, activation):

    l = inp
    for hid_size in hid_sizes:
        l = tf.contrib.layers.fully_connected(l, hid_size, activation)
    return l


class ActorCritic(object):
    def __init__(self, state_size, num_actions, distr='gauss_mu_sd',
                 hid_sizes_ac=(100, ), hid_sizes_cr=(100, ),
                 activation=tf.nn.elu, scope='actor_critic'):
        self.scope = scope
        self.distr_name = distr
        self.num_actions = num_actions

        with tf.variable_scope(scope):
            self.states = tf.placeholder(tf.float32, [None, state_size], "states")
            #self.states = tf.placeholder(tf.float32, [500, state_size], "states")

            with tf.variable_scope('critic'):
                crit_hid = build_fc_net(self.states, hid_sizes_cr, activation)
                val = tf.contrib.layers.fully_connected(crit_hid, 1, None)
                self.val = tf.reshape(val, [-1])

            with tf.variable_scope('actor'):
                policy_hid = build_fc_net(self.states, hid_sizes_ac, activation)
                #policy_out = tf.contrib.layers.fully_connected(policy_hid, num_actions * 2, None)
                #self.distr = DiagGaussianPd(policy_out)
                policy_out = tf.contrib.layers.fully_connected(policy_hid, num_actions, None)
                self.distr = BernoulliPd(policy_out)
                '''
                if distr == 'beta':
                    policy_out = tf.contrib.layers.fully_connected(policy_hid, num_actions*2, tf.nn.softplus) + 10e-7
                    alpha, beta = tf.split(policy_out, 2, axis=1)
                    self.distr = tf.contrib.distributions.Beta(alpha, beta)
                elif distr == 'gauss_mu_sd':
                    policy_out = tf.contrib.layers.fully_connected(policy_hid, num_actions * 2, None)
                    self.distr
                    mu, sigma = tf.split(policy_out, 2, axis=1)
                    sigma = tf.nn.softplus(sigma)
                    self.distr = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)
                elif distr == 'gauss_mu':
                    mu = tf.contrib.layers.fully_connected(policy_hid, num_actions, None)
                    sigma = tf.get_variable('sigma', [1, num_actions])
                    sigma = tf.nn.softplus(sigma)
                    self.distr = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)
                else:
                    raise ValueError('Unknown distribution')
                '''

                self.action_sample = self.distr.sample()
                self.action_mode = self.distr.mode()

        self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + '/actor')
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + '/critic')

    def act(self, state, sample=True, sess=None):
        sess = sess or tf.get_default_session()
        state = np.asarray([state]).astype('float32')

        a_tensor = self.action_mode
        if sample:
            a_tensor = self.action_sample

        a, v = sess.run([a_tensor, self.val], {self.states: state})
        return a.ravel()

    def value_batch(self, states, sess=None):
        sess = sess or tf.get_default_session()
        states = np.asarray(states, dtype='float32')
        return sess.run(self.val, {self.states: states})


class ActorCriticNumpy(object):
    def __init__(self, weights, activation, distr='gauss_mu_sd'):
        assert distr in ['gauss_mu_sd', 'gauss_mu', 'beta', 'bernoulli']

        self.activation = activation
        self.distr = distr
        self.set_weights(weights)

    def set_weights(self, weights):
        if self.distr == 'gauss_mu':
            self.weights = weights[:-1]
            self.sigma = softplus(weights[-1][0])
        else:
            self.weights = weights
            self.sigma = None
        self.num_layers = len(self.weights)/2

    def save_weights(self, fname='weights.pkl'):
        with open(fname, 'wb') as f:
            cPickle.dump(self.weights, f, -1)

    def act(self, s, sample=True):
        x = s
        for i in xrange(self.num_layers):
            x = np.dot(x, self.weights[2*i]) + self.weights[2*i+1]
            if i != self.num_layers - 1:
                x = self.activation(x)

        if self.distr == 'gauss_mu_sd':
            mu = x[:len(x)/2]
            log_std = np.clip(x[len(x)/2:], -1e3, -1.)
            std = np.exp(log_std)
            #std = np.exp(x[len(x) / 2:])
            a = gaussian(mu, std, sample)
        elif self.distr == 'gauss_mu':
            mu = x[:len(x) / 2]
            std = self.sigma
            a = gaussian(mu, std, sample)
        elif self.distr == 'beta':
            x = softplus(x)
            alpha = x[:len(x)/2]
            beta = x[len(x)/2:]
            if sample:
                a = np.random.beta(alpha, beta)
            else:
                a = [np.random.beta(alpha, beta) for _ in xrange(1000)]
                a = np.mean(a, axis=0)
        elif self.distr == 'bernoulli':
            probs = sigmoid(x)
            if sample:
                a = bernoulli.rvs(probs)
            else:
                a = np.round(probs)
        else:
            raise ValueError

        a = np.clip(a, 0., 1.)
        #a = np.clip(a, 0., 10.)
        #a = np.clip(a, -1, 1)
        return a


class PPO(object):
    def __init__(self, ac, ac_old, actor_lr=1e-4, crit_lr=2e-4,
                 clip_coeff=0.2, entropy_coeff=0.):
        self.ac = ac
        self.ac_old = ac_old

        self.actions = tf.placeholder(tf.float32, [None, ac.num_actions], 'actions')
        self.adv = tf.placeholder(tf.float32, [None], 'advantage')
        self.val_target = tf.placeholder(tf.float32, [None], 'val_target')
        #self.check = tf.add_check_numerics_ops()

        # actor loss
        ratio = tf.exp(ac.distr.logp(self.actions) - ac_old.distr.logp(self.actions))
        ratio = tf.where(tf.is_nan(ratio), tf.zeros_like(ratio), ratio)
        ratio = tf.where(tf.is_inf(ratio), tf.zeros_like(ratio), ratio)
        ratio = tf.clip_by_value(ratio, -1e3, 1e3)
        if self.ac.distr_name == 'beta':
            ratio = tf.reduce_prod(ratio, axis=1)
        ratio_clip = tf.clip_by_value(ratio, 1. - clip_coeff, 1. + clip_coeff)
        actor_obj = tf.reduce_mean(tf.minimum(ratio*self.adv, ratio_clip*self.adv))
        entropy = tf.reduce_mean(ac.distr.entropy())
        actor_loss = -1.*(actor_obj + entropy_coeff*entropy)
        optimizer = tf.train.AdamOptimizer(actor_lr)
        ac_grads = tf.gradients(actor_loss, ac.actor_params)
        ac_grads = [tf.where(tf.is_inf(g), tf.zeros_like(g), g) for g in ac_grads]
        ac_grads = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in ac_grads]
        ac_grads, _ = tf.clip_by_global_norm(ac_grads, 10)
        ac_grads_vars = zip(ac_grads, ac.actor_params)
        self.actor_update = optimizer.apply_gradients(ac_grads_vars)
        #self.actor_update = optimizer.minimize(actor_loss, var_list=ac.actor_params)

        # critic loss
        crit_loss = 0.5 * tf.square(ac.val - self.val_target)
        crit_loss = tf.reduce_mean(crit_loss)
        optimizer = tf.train.AdamOptimizer(crit_lr)
        cr_grads = tf.gradients(crit_loss, ac.critic_params)
        cr_grads = [tf.where(tf.is_inf(g), tf.zeros_like(g), g) for g in cr_grads]
        cr_grads = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in cr_grads]
        cr_grads, _ = tf.clip_by_global_norm(cr_grads, 10)
        cr_grads_vars = zip(cr_grads, ac.critic_params)
        self.crit_update = optimizer.apply_gradients(cr_grads_vars)
        #grad_vars_critic = optimizer.compute_gradients(crit_loss, var_list=self.ac.critic_params)
        #self.crit_update = optimizer.minimize(crit_loss, var_list=self.ac.critic_params)

        # update old policy to new policy
        self.sync = [v1.assign(v2) for v1, v2 in zip(ac_old.actor_params, ac.actor_params)]

        # add summaries
        with tf.name_scope('actor'):
            tf.summary.scalar('objective', actor_obj)
            tf.summary.scalar('ratio', tf.reduce_mean(ratio))
            tf.summary.scalar('entropy', entropy)
            #tf.summary.scalar('mean', tf.reduce_mean(ac.distr.mean))
            #tf.summary.scalar('std', tf.reduce_mean(ac.distr.std))
            tf.summary.scalar('probs', tf.reduce_mean(ac.distr.ps))

        with tf.name_scope('weights_grads_actor'):
            for g, v in ac_grads_vars:
                name = v.name
                name_grad = v.name + '_grad'
                tf.summary.scalar(name, tf.reduce_mean(tf.abs(v)))
                tf.summary.scalar(name_grad, tf.reduce_mean(tf.abs(g)))

        with tf.name_scope('critic'):
            tf.summary.scalar('loss', crit_loss)
            tf.summary.scalar('value', tf.reduce_mean(self.val_target))
        with tf.name_scope('weights_grads_critic'):
            for g, v in cr_grads_vars:
                name = v.name
                name_grad = v.name + '_grad'
                tf.summary.scalar(name, tf.reduce_mean(tf.abs(v)))
                tf.summary.scalar(name_grad, tf.reduce_mean(tf.abs(g)))
        self.summary = tf.summary.merge_all()

    def train_actor(self, states, actions, advantages, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.ac.states: states,
                     self.ac_old.states: states,
                     self.actions: actions,
                     self.adv: advantages,
                     #self.val_target: advantages
                     }
        #sess.run([self.check, self.actor_update], feed_dict=feed_dict)
        sess.run(self.actor_update, feed_dict=feed_dict)

    def train_critic(self, states, targets, sess):
        sess = sess or tf.get_default_session()
        feed_dict = {self.ac.states: states, self.val_target: targets}
        #sess.run([self.check, self.crit_update], feed_dict=feed_dict)
        sess.run(self.crit_update, feed_dict=feed_dict)

    def train(self, states, actions, targets, k, sess=None, summary_writer=None):
        sess = sess or tf.get_default_session()
        targets_pred = self.ac.value_batch(states, sess)
        advantages = targets - targets_pred
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        feed_dict = {self.ac.states: states,
                     self.ac_old.states: states,
                     self.actions: actions,
                     self.adv: advantages,
                     self.val_target: targets
                     }
        for _ in xrange(k):
            #self.train_actor(states, actions, advantages, sess)
            #self.train_critic(states, targets, sess)
            _, _, summary = sess.run([self.actor_update, self.crit_update, self.summary],
                                     feed_dict=feed_dict)
            if summary_writer is not None:
                summary_writer.add_summary(summary)

        self.update_old_ac(sess)

    def update_old_ac(self, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.sync)


class PolicyGradient(object):
    def __init__(self, ac, actor_lr=1e-4, crit_lr=5e-4, entropy_coeff=0.001):
        self.ac = ac

        self.actions = tf.placeholder(tf.float32, [None, ac.num_actions], 'actions')
        self.adv = tf.placeholder(tf.float32, [None], 'advantage')
        self.val_target = tf.placeholder(tf.float32, [None], 'val_target')

        # actor loss
        log_prob = ac.distr.logp(self.actions)
        #if self.ac.distr_name == 'beta':
        #    log_prob = tf.reduce_sum(log_prob, axis=1)
        actor_obj = tf.reduce_mean(log_prob*self.adv)
        entropy = tf.reduce_mean(ac.distr.entropy())
        actor_loss = -1.*(actor_obj + entropy_coeff*entropy)
        optimizer = tf.train.AdamOptimizer(actor_lr)
        #self.actor_update = optimizer.minimize(actor_loss, var_list=ac.actor_params)
        ac_grads = tf.gradients(actor_loss, ac.actor_params)
        ac_grads = [tf.where(tf.is_inf(g), tf.zeros_like(g), g) for g in ac_grads]
        ac_grads = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in ac_grads]
        ac_grads, _ = tf.clip_by_global_norm(ac_grads, 10)
        ac_grads_vars = zip(ac_grads, ac.actor_params)
        self.actor_update = optimizer.apply_gradients(ac_grads_vars)

        # critic loss
        crit_loss = 0.5 * tf.square(ac.val - self.val_target)
        crit_loss = tf.reduce_mean(crit_loss)
        optimizer = tf.train.AdamOptimizer(crit_lr)
        self.crit_update = optimizer.minimize(crit_loss, var_list=self.ac.critic_params)
        cr_grads = tf.gradients(crit_loss, ac.critic_params)
        cr_grads = [tf.where(tf.is_inf(g), tf.zeros_like(g), g) for g in cr_grads]
        cr_grads = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in cr_grads]
        cr_grads, _ = tf.clip_by_global_norm(cr_grads, 10)
        cr_grads_vars = zip(cr_grads, ac.critic_params)
        self.crit_update = optimizer.apply_gradients(cr_grads_vars)

        # add summaries
        with tf.name_scope('actor'):
            tf.summary.scalar('objective', actor_obj)
            tf.summary.scalar('entropy', entropy)
            #tf.summary.scalar('mean', tf.reduce_mean(ac.distr.mean))
            #tf.summary.scalar('std', tf.reduce_mean(ac.distr.std))
            tf.summary.scalar('probs', tf.reduce_mean(ac.distr.ps))

        with tf.name_scope('weights_grads_actor'):
            for g, v in ac_grads_vars:
                name = v.name
                name_grad = v.name + '_grad'
                tf.summary.scalar(name, tf.reduce_mean(tf.abs(v)))
                tf.summary.scalar(name_grad, tf.reduce_mean(tf.abs(g)))

        with tf.name_scope('critic'):
            tf.summary.scalar('loss', crit_loss)
            tf.summary.scalar('value', tf.reduce_mean(self.val_target))
        with tf.name_scope('weights_grads_critic'):
            for g, v in cr_grads_vars:
                name = v.name
                name_grad = v.name + '_grad'
                tf.summary.scalar(name, tf.reduce_mean(tf.abs(v)))
                tf.summary.scalar(name_grad, tf.reduce_mean(tf.abs(g)))
        self.summary = tf.summary.merge_all()

    def train_actor(self, states, action, advantages, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.ac.states: states,
                     self.actions: action,
                     self.adv: advantages,
                     }
        sess.run(self.actor_update, feed_dict)

    def train_critic(self, states, targets, sess):
        sess = sess or tf.get_default_session()
        feed_dict = {self.ac.states: states, self.val_target: targets}
        sess.run(self.crit_update, feed_dict)

    def train(self, states, actions, targets, sess=None, summary_writer=None):
        sess = sess or tf.get_default_session()
        targets_pred = self.ac.value_batch(states, sess)
        advantages = targets - targets_pred
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #self.train_actor(states, actions, advantages, sess)
        #self.train_critic(states, targets, sess)
        feed_dict = {self.ac.states: states,
                     self.actions: actions,
                     self.adv: advantages,
                     self.val_target: targets
                     }

        _, _, summary = sess.run([self.actor_update, self.crit_update, self.summary],
                                 feed_dict=feed_dict)
        if summary_writer is not None:
            summary_writer.add_summary(summary)
