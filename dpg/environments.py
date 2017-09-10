import numpy as np
from osim.env import RunEnv


class RunEnv2(RunEnv):
    def __init__(self, state_transform, visualize=False, max_obstacles=0):
        super(RunEnv2, self).__init__(visualize, max_obstacles)
        self.state_transform = state_transform

    def reset(self, difficulty=2, seed=None):
        s = super(RunEnv2, self).reset(difficulty=difficulty, seed=seed)
        self.state_transform.reset()
        s, _ = self.state_transform.process(s)
        return s

    def _step(self, action):
        s, r, t, info = super(RunEnv2, self)._step(action)
        info['original_state'] = s
        info['original_reward'] = r
        s, obst_rew = self.state_transform.process(s)
        return s, (r+obst_rew)*100, t, info


class JumpEnv(RunEnv):
    noutput = 9
    ninput = 38

    def __init__(self, visualize=False, max_obstacles=0):
        super(JumpEnv, self).__init__(visualize, max_obstacles)

    def get_observation(self):
        observation = super(JumpEnv, self).get_observation()
        return observation[:-3]

    def _step(self, action):
        action = np.tile(action, 2)
        #action = np.repeat(action, 2)
        s, r, t, info = super(JumpEnv, self)._step(action)
        return s, 10*r, t, info