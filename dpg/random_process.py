from __future__ import division
import numpy as np


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self, sigma=None):
        if sigma is None:
            sigma = self.current_sigma
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class RandomActivation(object):
    def __init__(self, size=18, reps_min=1, reps_max=3, min_miscles=1, max_muscles=None):
        self.size = size
        self.reps_min = reps_min
        self.reps_max = reps_max
        self.min_miscles = min_miscles
        self.max_muscles = size if max_muscles is None else min(size, max_muscles)
        self.all_muscles = np.arange(size)
        self.x = np.zeros(18)
        self.counter = 0

    def sample(self):
        if self.counter == 0:
            self.counter = np.random.randint(self.reps_min, self.reps_max+1)
            num_muscles = np.random.randint(self.min_miscles, self.max_muscles+1)
            muscles = np.random.choice(self.all_muscles, num_muscles, replace=False)
            self.x.fill(0)
            self.x[muscles] = 1

        self.counter -= 1
        return self.x

    def reset_states(self):
        self.counter = 0
        self.x.fill(0)
