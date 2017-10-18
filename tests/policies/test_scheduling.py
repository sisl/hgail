
import numpy as np
import unittest

from hgail.policies.scheduling import ConstantIntervalScheduler

class TestConstantIntervalScheduler(unittest.TestCase):

    def test_constant_interval_scheduler(self):
        # k = 2
        scheduler = ConstantIntervalScheduler(k=2)
        scheduler.reset(dones=[True, True, True])
        indicators = scheduler.should_update(observations=None)
        np.testing.assert_array_equal(indicators, [True] * 3)
        scheduler.reset(dones=[True, False, True])
        indicators = scheduler.should_update(observations=None)
        np.testing.assert_array_equal(indicators, [True, False, True])
        scheduler.reset(dones=[False, False, False])
        indicators = scheduler.should_update(observations=None)
        np.testing.assert_array_equal(indicators, [False, True, False])

        # k = inf
        scheduler = ConstantIntervalScheduler()
        scheduler.reset(dones=[True, True, True])
        indicators = scheduler.should_update(observations=None)
        np.testing.assert_array_equal(indicators, [True, True, True])
        for _ in range(10):
            indicators = scheduler.should_update(observations=None)
        np.testing.assert_array_equal(indicators, [False, False, False])

if __name__ == '__main__':
    unittest.main()