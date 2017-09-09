import numpy as np


def get_state_names(all=False):
    names = ['pelvis_' + n for n in ('rot', 'x', 'y')]
    names += ['pelvis_vel_' + n for n in ('rot', 'x', 'y')]
    names += ['hip_right', 'knee_right', 'ankle_right', 'hip_left', 'knee_left', 'ankle_left']
    names += ['hip_right_vel', 'knee_right_vel', 'ankle_right_vel', 'hip_left_vel', 'knee_left_vel', 'ankle_left_vel']
    names += ['mass_x', 'mass_y']
    names += ['mass_x_vel', 'mass_y_vel']

    if all:
        names += [b + '_' + i for b in ['head', 'pelvis2', 'torso', 'toes_left',
                                        'toes_right', 'talus_left', 'talus_right'] for i in
                  ['x', 'y']]
    else:
        names += [b + '_' + i for b in ['head', 'torso', 'toes_left', 'toes_right',
                                        'talus_left', 'talus_right'] for i in
                  ['x', 'y']]

    names += ['muscle_left', 'muscle_right']
    names += ['obst_dist', 'obst_y', 'obst_r']
    return names


def get_names_to_center():
    return [b + '_x' for b in ['head', 'mass', 'torso', 'toes_left',
                               'toes_right', 'talus_left', 'talus_right']]

def get_names_obstacles():
    return [b + '_x' for b in ['toes_left', 'toes_right', 'talus_left', 'talus_right']]


def get_names_vel():
    return [b + '_' + i for b in ['head', 'torso', 'toes_left', 'toes_right', 'talus_left', 'talus_right']
            for i in ['x', 'y']]


def calculate_velocity(cur, prev):
    if prev is None:
        return np.zeros_like(cur, dtype=np.float32)
    return 100.*(cur - prev)


def _get_pattern_idxs(lst, pattern):
    idxs = [i for i, x in enumerate(lst) if pattern in x]
    return idxs


class State(object):
    def __init__(self, exclude_obstacles=False):
        self.exclude_obstacles = exclude_obstacles
        self.state_idxs = [i for i, n in enumerate(get_state_names(True)) if n not in ['pelvis2_x', 'pelvis2_y']]
        self.state_names = get_state_names()

        if exclude_obstacles:
            self.state_names = self.state_names[:-3]
        else:
            names_obst = get_names_obstacles()
            self.obst_idxs = [self.state_names.index(n) for n in names_obst]
            self.state_names += [n + '_obst' for n in names_obst]
        self._set_left_right()

    def _set_left_right(self):
        self.left_idxs = _get_pattern_idxs(self.state_names, '_left')
        self.right_idxs = _get_pattern_idxs(self.state_names, '_right')

    def reset(self):
        pass

    def process(self, state):
        state = np.asarray(state, dtype=np.float32)
        state = state[self.state_idxs]
        state[-3] = 10 if state[-3]>10 else state[-3]
        obst_x = state[-3] + state[1]
        obst_dist = [obst_x - state[i] for i in self.obst_idxs]

        if self.exclude_obstacles:
            return state[:-3]
        else:
            state = np.concatenate([state, obst_dist])
        
        return state

    def flip_state(self, state, copy=True):
        assert np.ndim(state) == 1
        state = np.asarray(state, dtype=np.float32)
        state = self.flip_states(state.reshape(1, -1), copy)
        return state.ravel()

    def flip_states(self, states, copy=True):
        assert np.ndim(states) == 2
        states = np.asarray(states, dtype=np.float32)
        if copy:
            states = states.copy()
        left = states[:, self.left_idxs]
        right = states[:, self.right_idxs]
        states[:, self.left_idxs] = right
        states[:, self.right_idxs] = left
        return states

    @property
    def state_size(self):
        return len(self.process(np.zeros(41, dtype=np.float32)))


class StateVel(State):
    def __init__(self, vel_states=get_names_vel(), exclude_obstacles=False):
        super(StateVel, self).__init__(exclude_obstacles)
        self.vel_idxs = [self.state_names.index(k) for k in vel_states]
        self.prev_vals = None
        self.state_names += [n + '_vel' for n in vel_states]
        # left right idxs
        self._set_left_right()

    def reset(self):
        self.prev_vals = None

    def process(self, state):
        state = super(StateVel, self).process(state)
        cur_vals = state[self.vel_idxs]
        vel = calculate_velocity(cur_vals, self.prev_vals)
        self.prev_vals = cur_vals
        state = np.concatenate((state, vel))
        return state


class StateVelCentr(State):
    def __init__(self, centr_state='pelvis_x', states_to_center=get_names_to_center(),
                 vel_states=get_names_vel(), vel_before_centr=True,
                 exclude_obstacles=False):
        super(StateVelCentr, self).__init__(exclude_obstacles)

        # center
        self.centr_idx = self.state_names.index(centr_state)
        self.states_to_center = [self.state_names.index(k) for k in states_to_center]
        # velocities
        self.prev_vals = None
        self.vel_idxs = [self.state_names.index(k) for k in vel_states]
        self.vel_before_centr = vel_before_centr
        self.state_names += [n + '_vel' for n in vel_states]
        # left right idxs
        self._set_left_right()

    def reset(self):
        self.prev_vals = None

    def process(self, state):
        state = super(StateVelCentr, self).process(state)

        if self.vel_before_centr:
            cur_vals = state[self.vel_idxs]
            vel = calculate_velocity(cur_vals, self.prev_vals)
            self.prev_vals = cur_vals
            state[self.states_to_center] -= state[self.centr_idx]
        else:
            state[self.states_to_center] -= state[self.centr_idx]
            cur_vals = state[self.vel_idxs]
            vel = calculate_velocity(cur_vals, self.prev_vals)
            self.prev_vals = cur_vals

        state = np.concatenate((state, vel))
        return state
