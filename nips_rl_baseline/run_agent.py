import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import Callback
from env import RunEnv3
import argparse


# Command line parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('--test', action='store_true', help='Run test episodes')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--test_period', type=int, default=100, help='testing period in episodes')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--visualize', action='store_true', help='Run training with visualization.')
    parser.add_argument('--model', type=str, default="models/actor_model.h5", help='Filename with model.')
    return parser.parse_args()


def build_actor(n_state, n_actions):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1, n_state)))
    actor.add(Dense(64))
    actor.add(Activation('elu'))
    actor.add(Dense(64))
    actor.add(Activation('elu'))
    actor.add(Dense(n_actions))
    actor.add(Activation('sigmoid'))
    print(actor.summary())
    return actor


def build_critic(n_state, n_actions):
    action_input = Input(shape=(n_actions,), name='action_input')
    observation_input = Input(shape=(1, n_state), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(32)(x)
    x = Activation('elu')(x)
    x = Dense(32)(x)
    x = Activation('elu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())
    return critic, action_input


def test_env(env, actor, num_episodes=3):
    total_reward = 0
    for _ in xrange(num_episodes):
        state = env.reset()
        terminal = False
        while not terminal:
            state = np.asarray([[state]])
            a = actor.predict(state)[0]
            state, reward, terminal, _ = env.step(a)
            total_reward += reward

    mean_reward = 1. * total_reward / 3.
    print('test reward {:.4f}'.format(mean_reward))
    return mean_reward


class SaveActorCallback(Callback):
    def __init__(self, test_period=100, fname='actor_model.h5'):
        self.test_period = test_period
        self.episodes = 0
        self.best_reward = -1e8
        self.fname = fname

    def on_episode_end(self, episode, logs={}):
        self.episodes += 1
        if self.episodes % self.test_period == 0:
            mean_reward = test_env(self.env, self.model.actor, 3)
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.actor.save(self.fname)


def train(fname_model, steps, gamma,  test_period, visualize=False):
    env = RunEnv3(visualize)
    n_actions = env.noutput
    n_state = env.observation_space.shape[0]

    actor = build_actor(n_state, n_actions)
    critic, action_input = build_critic(n_state, n_actions)

    # Set up the agent for training
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.1, size=n_actions,
                                              sigma_min=1e-6, n_steps_annealing=1e6)

    agent = DDPGAgent(nb_actions=n_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=gamma, target_model_update=1e-3,
                      delta_clip=1.)
    agent.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])

    try:
        save_callback = SaveActorCallback(test_period, fname_model)
        agent.fit(env, nb_steps=steps, visualize=False, verbose=1,
                  nb_max_episode_steps=env.timestep_limit,
                  log_interval=10000, callbacks=[save_callback])
    finally:
        actor.save(fname_model + '_last')


def test(fname_model, visualize):
    env = RunEnv3(visualize)
    actor = load_model(fname_model)
    test_env(env, actor, 3)


if __name__ == '__main__':
    args = get_args()

    if not args.test:
        train(args.model, args.steps, args.gamma, args.test_period, args.visualize)
    else:
        test(args.model, args.visualize)
