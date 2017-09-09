import numpy as np
from sklearn.linear_model import LinearRegression

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
    return [b + '_x' for b in ['head', 'pelvis', 'torso', 'toes_left',
                               'toes_right', 'talus_left', 'talus_right']]


def get_bodies_names():
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
    def __init__(self, exclude_obstacles=False, last_n_bodies=3):
        self.exclude_obstacles = exclude_obstacles
        self.state_idxs = [i for i, n in enumerate(get_state_names(True)) if n not in ['pelvis2_x', 'pelvis2_y']]
        self.state_names = get_state_names()
        self.step = 0

        if exclude_obstacles:
            self.state_names = self.state_names[:-3]

        self.predict_bodies = last_n_bodies > 0
        self.last_n_bodies = last_n_bodies
        self.bodies_idxs = [self.state_names.index(n) for n in get_bodies_names()]
        if self.predict_bodies:
            # 2 last dimensions, first for emulator values, second for predicted
            self.last_bodies = np.zeros(shape=(1000, len(self.bodies_idxs), 2))
            self._x = np.arange(self.last_n_bodies).reshape(-1, 1)
            self._x_pred = np.asarray([[self.last_n_bodies]])
            self._reg = LinearRegression()
            self.bodies_flt = np.zeros(len(self.state_names), dtype='bool')
            self.bodies_flt[self.bodies_idxs] = 1

        self.state_names_out = self.state_names
        self._set_left_right()

    def _set_left_right(self):
        self.left_idxs = _get_pattern_idxs(self.state_names, '_left')
        self.right_idxs = _get_pattern_idxs(self.state_names, '_right')

    def reset(self):
        self.step = 0

    def _predict_bodies(self, state):
        #print 'state before', state
        self._update_bodies(state, 0)

        # if enough steps check if prediction is needed
        if self.step >= self.last_n_bodies:
            bodies_predict_flt = self.last_bodies[self.step, :, 0] == self.last_bodies[self.step-1, :, 0]

            if np.any(bodies_predict_flt):
                _state_bodies = state[self.bodies_flt]
                #print '\npredicting', self.step
                #print 'current vals', _state_bodies[bodies_predict_flt]
                y = self.last_bodies[self.step - self.last_n_bodies:self.step, bodies_predict_flt, 1]
                #_y = self.last_bodies[self.step - self.last_n_bodies:self.step, bodies_predict_flt, 0]
                #print 'last_vals', _y
                self._reg.fit(self._x, y)
                y_pred = self._reg.predict(self._x_pred)[0]
                #print 'predicted vals', y_pred

                _state_bodies[bodies_predict_flt] = y_pred
                state[self.bodies_flt] = _state_bodies
                #print 'state after', state[self.bodies_idxs]

    def _update_bodies(self, state, axis):
        if self.predict_bodies:
            self.last_bodies[self.step, :, axis] = state[self.bodies_idxs]

    def process(self, state):
        state = np.asarray(state, dtype=np.float32)
        state = state[self.state_idxs]
        state[-3] = np.exp(-1.*state[-3])
        if self.exclude_obstacles:
            return state[:-3]

        # update last bodies
        #state_no_pred = state.copy()
        if self.predict_bodies:
            self._predict_bodies(state)
            #state_out = (state_no_pred + state)/2
            #state_out = state
            self._update_bodies(state, 1)
        #else:
        #    state_out = state_no_pred

        self.step += 1
        #return (state_no_pred + state)/2.
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
    def __init__(self, vel_states=get_bodies_names(), exclude_obstacles=False, last_n_bodies=3):
        super(StateVel, self).__init__(exclude_obstacles, last_n_bodies)
        self.vel_idxs = [self.state_names.index(k) for k in vel_states]
        self.prev_vals = None
        self.state_names += [n + '_vel' for n in vel_states]
        self.state_names_out = self.state_names
        # left right idxs
        self._set_left_right()

    def reset(self):
        super(StateVel, self).reset()
        self.prev_vals = None

    def process(self, state):
        state = super(StateVel, self).process(state)
        cur_vals = state[self.vel_idxs]
        vel = calculate_velocity(cur_vals, self.prev_vals)
        self.prev_vals = cur_vals
        state = np.concatenate((state, vel))
        return state


class StateVelCentr(State):
    def __init__(self, centr_state='mass_x', states_to_center=get_names_to_center(),
                 vel_states=get_bodies_names(), vel_before_centr=True,
                 exclude_obstacles=False, exclude_centr=False, last_n_bodies=3):
        super(StateVelCentr, self).__init__(exclude_obstacles, last_n_bodies)

        # center
        self.centr_idx = self.state_names.index(centr_state)
        self.states_to_center = [self.state_names.index(k) for k in states_to_center]
        # velocities
        self.prev_vals = None
        self.vel_idxs = [self.state_names.index(k) for k in vel_states]
        self.vel_before_centr = vel_before_centr
        self.state_names += [n + '_vel' for n in vel_states]
        self.exclude_centr = exclude_centr

        if self.exclude_centr:
            self.state_names_out = self.state_names[:max(0, self.centr_idx)] + \
                          self.state_names[self.centr_idx + 1:]
        else:
            self.state_names_out = self.state_names

        # left right idxs
        self._set_left_right()

    def _set_left_right(self):
        state_names = self.state_names_out
        self.left_idxs = _get_pattern_idxs(state_names, '_left')
        self.right_idxs = _get_pattern_idxs(state_names, '_right')

    def reset(self):
        super(StateVelCentr, self).reset()
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

        if self.exclude_centr:
            state = np.concatenate([state[:max(0, self.centr_idx)], state[self.centr_idx+1:]])

        state = np.concatenate((state, vel))
        return state
