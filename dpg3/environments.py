import numpy as np
from osim.env import RunEnv
from gym.spaces import Box, MultiBinary


class RunEnv2(RunEnv):
    def __init__(self, state_transform, visualize=False, max_obstacles=10,
                 skip_frame=5, reward_mult=10., fall_penalty=0):
        super(RunEnv2, self).__init__(visualize, max_obstacles)
        self.state_transform = state_transform
        self.observation_space = Box(-1000, 1000, state_transform.state_size)
        self.action_space = MultiBinary(18)
        self.skip_frame = skip_frame
        self.reward_mult = reward_mult
        self.fall_penalty = fall_penalty

    def reset(self, difficulty=2, seed=None):
        s = super(RunEnv2, self).reset(difficulty=difficulty, seed=seed)
        self.state_transform.reset()
        s, _ = self.state_transform.process(s)
        return s

    def _step(self, action):
        action = np.clip(action, 0, 1)
        info = {'original_reward': 0}
        reward = 0.
        for _ in range(self.skip_frame):
            s, r, t, _ = super(RunEnv2, self)._step(action)
            info['original_reward'] += r
            s, obst_rew = self.state_transform.process(s)
            reward += r + obst_rew
            if t:
                break

        if self.is_pelvis_too_low():
            reward += self.fall_penalty

        return s, reward*self.reward_mult, t, info


class JumpEnv(RunEnv):
    noutput = 9
    ninput = 38

    def __init__(self, visualize=False, max_obstacles=0):
        super(JumpEnv, self).__init__(visualize, max_obstacles)
        self.action_space = MultiBinary(9)

    def get_observation(self):
        observation = super(JumpEnv, self).get_observation()
        return observation[:-3]

    def _step(self, action):
        action = np.tile(action, 2)
        #action = np.repeat(action, 2)
        s, r, t, info = super(JumpEnv, self)._step(action)
        return s, 10*r, t, info
