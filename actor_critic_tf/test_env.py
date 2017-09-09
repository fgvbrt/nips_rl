import gym
from gym.spaces import Box
import numpy as np
import sklearn
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline, FeatureUnion


class EnvFourierFeatures(object):
    def __init__(self, env):
        self.env = env
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        featurizer.fit(scaler.transform(observation_examples))
        self.pipeline = Pipeline([('skaler', scaler), ('rbf', featurizer)])
        self.observation_space = Box(-np.inf, np.inf, (400,))

    def reset(self):
        s = self.env.reset()
        return self.pipeline.transform([s])[0]

    def step(self, action):
        s, a, r, t = self.env.step(action)
        s = self.pipeline.transform([s])[0]
        return s, a, r, t

    @property
    def action_space(self):
        return self.env.action_space


def make_test_env(fourier_features=True):
    env = gym.envs.make("MountainCarContinuous-v0")

    if fourier_features:
        return EnvFourierFeatures(env)
    else:
        return env
