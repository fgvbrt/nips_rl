import numpy as np


class ReplayMemory(object):
    def __init__(self, state_size, action_size, max_steps=100000, seed=None):
        self.max_steps = max_steps
        self.state_size = state_size
        self.action_size = action_size
        self.bottom = 0
        self.top = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

        # init buffers
        self.states = np.zeros(shape=(max_steps, state_size), dtype=np.float32)
        self.actions = np.zeros(shape=(max_steps, action_size), dtype=np.float32)
        self.rewards = np.zeros(max_steps, dtype=np.float32)
        self.terminals = np.zeros(max_steps, dtype='bool')

    def reset(self):
        self.bottom = 0
        self.top = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_sample(self, state, action, reward, terminal):
        """Add a time step record.
        Arguments:
            state -- observed state
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.states[self.top] = state
        self.terminals[self.top] = terminal
        self.actions[self.top] = action
        self.rewards[self.top] = reward

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def add_samples(self, states, actions, rewards, terminals):
        """Add a time step record.
        Arguments:
            state -- observed state
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        # easy part
        n = states.shape[0]
        idxs = range(self.top, self.top + n)
        self.terminals.put(idxs, terminals, mode='wrap')
        self.rewards.put(idxs, rewards, mode='wrap')

        # for states and action need to calculate idxs
        states_start_idx = self.top * self.state_size
        states_end_idx = states_start_idx + np.prod(states.shape)
        states_idxs = range(states_start_idx, states_end_idx)
        self.states.put(states_idxs, states, mode='wrap')

        act_start_idx = self.top * self.action_size
        act_end_idx = act_start_idx + np.prod(actions.shape)
        act_idxs = range(act_start_idx, act_end_idx)
        self.actions.put(act_idxs, actions, mode='wrap')

        if self.size == self.max_steps:
            self.bottom = (self.bottom + n) % self.max_steps
        else:
            self.size += n
            # one more check
            if self.size >= self.max_steps:
                d = self.size - self.max_steps
                self.size = self.max_steps
                self.bottom = (self.bottom + d) % self.max_steps
        self.top = (self.top + n) % self.max_steps

    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, and
            next_states for batch_size randomly chosen state transitions.
        """
        # Allocate the response.
        states = np.zeros((batch_size, self.state_size), dtype=self.states.dtype)
        actions = np.zeros((batch_size, self.action_size), dtype=self.actions.dtype)
        rewards = np.zeros((batch_size, 1), dtype=self.rewards.dtype)
        next_states = np.zeros_like(states, dtype=self.states.dtype)
        terminals = np.zeros((batch_size, 1), dtype=self.terminals.dtype)

        # uniform sampling
        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - 1)

            # check for terminal state
            if self.terminals.take(index, axis=0, mode='wrap') > 0:
                continue

            # Add the state transition to the response.
            states[count] = self.states.take(index, axis=0, mode='wrap')
            actions[count] = self.actions.take(index, axis=0, mode='wrap')
            rewards[count] = self.rewards.take(index, axis=0, mode='wrap')
            next_states[count] = self.states.take(index+1, axis=0, mode='wrap')
            terminals[count] = self.terminals.take(index+1, axis=0, mode='wrap')
            count += 1

        return states, actions, rewards, terminals, next_states

    def random_batch2(self, batch_size):
        """Return corresponding states, actions, rewards, and
            next_states for batch_size randomly chosen state transitions.
        """
        # shuld be more efficient variant of random sampling

        # Randomly choose a time step from the replay memory.
        index = self.rng.randint(self.bottom + batch_size,
                                 self.bottom + self.size - 1)
        idxs = np.arange(index-batch_size, index)
        # create mask for terminal stats
        m = ~self.terminals.take(idxs, axis=0, mode='wrap')

        # Add the state transition to the response.
        states = self.states.take(idxs, axis=0, mode='wrap')
        actions = self.actions.take(idxs, axis=0, mode='wrap')
        rewards = self.rewards.take(idxs, axis=0, mode='wrap').reshape(-1, 1)
        next_states = self.states.take(idxs+1, axis=0, mode='wrap')
        terminals = self.terminals.take(idxs+1, axis=0, mode='wrap').reshape(-1, 1)

        return states[m], actions[m], rewards[m], terminals[m], next_states[m]

    def load(self, filename):
        tmp_dict = np.load(filename)
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        tmp_dict = {k: v for k, v in self.__dict__.iteritems() if k is not 'rng'}
        np.savez(filename, **tmp_dict)
