
import numpy as np

from rllab.core.serializable import Serializable

class Scheduler(Serializable):

    def reset(self, dones=None):
        raise NotImplementedError

    def should_update(self, observations):
        raise NotImplementedError

class ConstantIntervalScheduler(Scheduler):

    def __init__(self, k=np.inf):
        Serializable.quick_init(self, locals())
        self.k = k
        self.counters = None

    def reset(self, dones):
        '''
        Reset internal values based on whether the respective environments have
        just terminated. Note that this function has no return values. This is 
        because the caller of this function should not update after it. It should
        update after calling should_update

        Args:
            - dones: list of booleans indicating terminal states of envs
        '''
        # if number of environments not previously set, then indicate to update
        if self.counters is None or len(self.counters) != len(dones):
            self.counters = np.zeros(len(dones))

        # always update after individual env reset
        for (i, done) in enumerate(dones):
            if done:
                self.counters[i] = 0

    def should_update(self, observations):
        '''
        Determines whether the caller should update based on obs.

        Args:
            observations: observations used to determine whether to update

        Returns:
            list of indicator values as to whether or not to update
        '''
        if self.counters is None:
            self.counters = np.zeros(len(observations))
        # if the episode step count evenly divides k, then update 
        # otherwise do not update
        indicators = self.counters % self.k == 0
        self.counters += 1
        return indicators