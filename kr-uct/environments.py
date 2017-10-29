import numpy as np
from osim.env import RunEnv
from gym.spaces import Box, MultiBinary
import opensim


class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 1e8


class RunEnv2(RunEnv):
    checkpoints = {}

    def __init__(self, visualize, max_obstacles):
        super(RunEnv2, self).__init__(visualize, max_obstacles)
        self.timestep_limit = 1e8
        self.spec = Spec()

    def clone_state(self):
        state = opensim.State(self.osim_model.state)
        self.checkpoints[state] = \
            [self.last_state[:], self.current_state[:], self.istep]
        return state

    def restore_state(self, state, delete=False):
        self.last_state, self.current_state, self.istep =\
            self.checkpoints[state]
        self.osim_model.state = opensim.State(state)

        if delete:
            self.checkpoints.pop(state, None)

    def clear_all_states(self):
        self.checkpoints = {}

    def _step(self, action):
        action = np.clip(action, 0, 1)
        s, r, t, info = super(RunEnv2, self)._step(action)
        s = np.round(s, 10)
        return s, r, t, info


class RunEnvStateTransform(RunEnv2):
    def __init__(self, state_transform, visualize=False, max_obstacles=3,
                 skip_frame=5, rev_scale=10):
        super(RunEnvStateTransform, self).__init__(visualize, max_obstacles)
        self.state_transform = state_transform
        self.observation_space = Box(-1000, 1000, state_transform.state_size)
        self.action_space = MultiBinary(18)
        self.skip_frame = skip_frame
        self.rev_scale = rev_scale

    def reset(self, difficulty=2, seed=None):
        s = super(RunEnvStateTransform, self).reset(difficulty=difficulty, seed=seed)
        self.state_transform.reset()
        s, _ = self.state_transform.process(s)
        return s

    def _step(self, action):
        info = {'original_reward':0}
        reward = 0.
        for _ in range(self.skip_frame):
            s, r, t, _ = super(RunEnvStateTransform, self)._step(action)
            info['original_reward'] += r
            s, obst_rew = self.state_transform.process(s)
            reward += r + obst_rew
            if t:
                break            

        return s, reward*self.rev_scale, t, info
