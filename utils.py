import numpy as np
import random
from collections import namedtuple, deque

class ExperienceMemory:
    """Experience Memory Buffer"""

    def __init__(self, buffer_size, batch_size):
        """Initialize buffer object."""
        self.memory = deque(maxlen=buffer_size)  # fifo queue
        self.batch_size = batch_size
        self.general_feeling = 0.0 # not used yet :)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.action_size = np.array(action).shape[0]
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sweet_dreams(self, batch_size=64):
        """Recall and replay experiences in memory, reordering experiences by positive reward."""
        for e in sorted(self.memory,key=lambda x: x.reward, reverse=False):
            self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def batch_samples(self, batch_size=64):
        """"""
        exp = self.sample(batch_size=64)
        states = np.vstack([e.state for e in exp if e is not None])
        actions = np.array([e.action for e in exp if e is not None]).astype(np.float32).reshape(-1, self.action_size)

        rewards = np.array([e.reward for e in exp if e is not None]).astype(np.float32).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in exp if e is not None])
        dones = np.array([e.done for e in exp if e is not None]).astype(np.uint8).reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self, mu=None):
        if mu != None:
            self.mu = mu * np.ones(self.size)
        self.state = self.mu
