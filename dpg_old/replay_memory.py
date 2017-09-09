import numpy as np


class ReplayMemory(object):
    """A replay memory consisting of circular buffers for observed states,
        actions, and rewards.
    """
    def __init__(self, state_size, action_size, max_steps=1000000):
        """Construct a ReplayMemory.
        Arguments:
            max_steps - the number of time steps to store
            rng - initialized numpy random number generator,
            used to choose random minibatches
        """

        # Store arguments.
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.rng = np.random.RandomState()

        # Allocate the circular buffers and indices.
        self.states = np.zeros((max_steps, self.state_size), dtype='float32')
        self.actions = np.zeros((max_steps, action_size), dtype='float32')
        self.rewards = np.zeros(max_steps, dtype='float32')
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return self.size

    def add_sample(self, state, terminal, action, reward):
        """Add a time step record.
        Arguments:
            state -- observed state
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.states[self.top] = state
        self.terminal[self.top] = terminal
        self.actions[self.top] = action
        self.rewards[self.top] = reward

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, and
            next_states for batch_size randomly chosen state transitions.
        """
        # Allocate the response.
        states = np.zeros((batch_size, self.state_size), dtype='float32')
        actions = np.zeros((batch_size, self.action_size), dtype='float32')
        rewards = np.zeros((batch_size, 1), dtype='float32')
        next_states = np.zeros((batch_size, self.state_size), dtype='float32')
        terminals = np.zeros((batch_size, 1), dtype='bool')

        # uniform sampling
        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - 1)

            if self.terminal.take(index, mode='wrap'):
                continue

            # Add the state transition to the response.
            states[count] = self.states.take(index, axis=0, mode='wrap')
            actions[count] = self.actions.take(index, axis=0, mode='wrap')
            rewards[count] = self.rewards.take(index, mode='wrap')
            next_states[count] = self.states.take(index+1, axis=0, mode='wrap')
            terminals[count] = self.terminal.take(index+1, mode='wrap')
            count += 1

        return states, actions, rewards, next_states, terminals

    def load(self, filename):
        tmp_dict = np.load(filename)
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        tmp_dict = {k: v for k, v in self.__dict__.iteritems() if k is not 'rng'}
        np.savez(filename, **tmp_dict)
