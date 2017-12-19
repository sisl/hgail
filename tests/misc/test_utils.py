
import copy
import numpy as np
import unittest

import hgail.misc.utils

class TestActionNormalizer(unittest.TestCase):

    def test_normalize(self):
        # does nothing with 0 mean 1 std dev
        n = hgail.misc.utils.ActionNormalizer(np.array([0,0]), np.array([1,1]))
        data = [[0,0],[1,1],[.5,.5],[-1,2],[2,-1]]
        data_copy = copy.deepcopy(data)
        actual = n.normalize(data_copy)

        expected = [[0,0],[1,1],[.5,.5],[-1,2],[2,-1]]
        np.testing.assert_array_almost_equal(expected, actual, 4)

        # check for data mutation
        np.testing.assert_array_equal(data, data_copy)

        # with nonzero mean nonone std dev
        n = hgail.misc.utils.ActionNormalizer(np.array([.5,1.5]), np.array([2,4]))
        expected = [np.subtract(v, [.5,1.5]) / [2,4] for v in data]
        actual = n.normalize(data_copy)
        np.testing.assert_array_almost_equal(expected, actual, 4)

class TestBatchToPathRewards(unittest.TestCase):

    def test_batch_to_path_rewards(self):

        lengths = [0, 1, 2]
        batch = [3, 4, 5]
        actual = hgail.misc.utils.batch_to_path_rewards(batch, lengths)
        expected = [[], [3], [4,5]]
        np.testing.assert_array_equal(expected, actual)

class TestRewardHandler(unittest.TestCase):

    def test_merge(self):

        reward_handler = hgail.misc.utils.RewardHandler(alpha=0.001)
        paths = [{'rewards': []}, {'rewards': [1]}, {'rewards': [1,2]}]
        critic_rewards = [[], [1], [1,2]]
        recognition_rewards = [[], [1], [1,2]]
        paths = reward_handler.merge(paths, critic_rewards, recognition_rewards)

        np.testing.assert_array_equal(paths[0]['rewards'], [])
        np.testing.assert_array_equal(paths[1]['rewards'], [3.])
        np.testing.assert_array_equal(paths[2]['rewards'], [3.,6.])

        reward_handler.critic_clip_low = 10
        reward_handler.critic_clip_high = 11
        reward_handler.critic_initial_scale = reward_handler.critic_final_scale = 2.
        reward_handler.recognition_initial_scale = reward_handler.recognition_final_scale = 5.

        paths = [{'rewards': []}, {'rewards': [1]}, {'rewards': [1,2]}]
        critic_rewards = [[], [1], [1,2]]
        recognition_rewards = [[], [1], [1,2]]
        paths = reward_handler.merge(paths, critic_rewards, recognition_rewards)

        np.testing.assert_array_equal(paths[0]['rewards'], [])
        np.testing.assert_array_equal(paths[1]['rewards'], [26.])
        np.testing.assert_array_equal(paths[2]['rewards'], [26.,32.])

        reward_handler.critic_initial_scale = 1.
        reward_handler.critic_final_scale = 2.
        reward_handler.recognition_initial_scale = 4.
        reward_handler.recognition_final_scale = 5.
        reward_handler.max_epochs = 10
        reward_handler.step = 0
        for _ in range(reward_handler.max_epochs):
            reward_handler.merge(paths, critic_rewards, recognition_rewards)
        self.assertAlmostEqual(reward_handler.critic_scale, 2.)
        self.assertAlmostEqual(reward_handler.recognition_scale, 5.)

    def test_update_reward_estimate(self):
        np.random.seed(1)
        reward_handler = hgail.misc.utils.RewardHandler(normalize_rewards=True, alpha=0.001)

        n_samples, timesteps = 30000, 50
        rewards = np.random.randn(n_samples, timesteps) + 5
        for r in rewards:
            reward_handler._update_reward_estimate(r, 'critic')
        self.assertTrue(np.abs(reward_handler.critic_reward_mean - 5) < 1e-2)
        self.assertTrue(np.abs(reward_handler.critic_reward_var - 1) < 1e-2)

    def test_normalize_rewards(self):
        np.random.seed(1)
        reward_handler = hgail.misc.utils.RewardHandler(normalize_rewards=True, alpha=0.001)

        n_samples, timesteps = 2000, 100
        rewards = np.random.randn(n_samples, timesteps)
        for r in rewards:
            reward_handler._normalize_rewards(r, 'critic')

        n = reward_handler._normalize_rewards(rewards.reshape(-1), 'critic')
        self.assertTrue(np.abs(np.mean(n)) < 1e-2)
        self.assertTrue(np.abs(np.std(n) - 1) < 1e-2)

    def test_normalize_rewards_critic_and_recognition(self):
        np.random.seed(1)
        reward_handler = hgail.misc.utils.RewardHandler(normalize_rewards=True, alpha=0.001)

        n_samples, timesteps = 20000, 100
        critic_rewards = np.random.randn(n_samples, timesteps)
        for r in critic_rewards:
            reward_handler._normalize_rewards(r, 'critic')
        n = reward_handler._normalize_rewards(critic_rewards.reshape(-1), 'critic')
        self.assertTrue(np.abs(np.mean(n)) < 1e-2)
        self.assertTrue(np.abs(np.std(n) - 1) < 1e-2)

        recognition_rewards = np.random.randn(n_samples, timesteps) - 1
        for r in recognition_rewards:
            reward_handler._normalize_rewards(r, 'recognition')

        # check critic not impacted
        new_n = reward_handler._normalize_rewards(critic_rewards.reshape(-1), 'critic')
        np.testing.assert_array_almost_equal(new_n, n, 4)
        n = reward_handler._normalize_rewards(recognition_rewards.reshape(-1), 'recognition')
        self.assertTrue(np.abs(np.mean(n) + 1)  < 1e-1)
        self.assertTrue(np.abs(np.std(n) - 1) < 1e-1)

class TestReplayMemory(unittest.TestCase):

    def test_replay_memory(self):

        # simple testing in adding and sampling with batch size 1
        mem = hgail.misc.utils.ReplayMemory(maxsize=2)
        paths = [1]
        mem.add(paths)
        np.testing.assert_array_equal(mem.mem, paths)

        paths = [2,3]
        mem.add(paths)
        np.testing.assert_array_equal(mem.mem, paths)

        paths = [4,5,6]
        mem.add(paths)
        np.testing.assert_array_equal(mem.mem, paths[-2:])

        paths = [6]
        mem.add(paths)
        sample = mem.sample(1)
        np.testing.assert_array_equal(sample, [6])

        # large batch size
        mem = hgail.misc.utils.ReplayMemory(maxsize=200)

        paths = [1] * 50
        mem.add(paths)
        sample = mem.sample(10)
        self.assertEqual(len(sample), 10)

        paths = [1] * 500
        mem.add(paths)
        self.assertEqual(len(mem.mem), 200)

class TestKeyValueReplayMemory(unittest.TestCase):

    def test_key_value_replay_memory(self):
        keys = ['x','a']

        # simple testing in adding and sampling with batch size 1
        mem = hgail.misc.utils.KeyValueReplayMemory(maxsize=2)
        values = dict(x=np.array([[1]]), a=np.array([[1]]))
        mem.add(keys, values)
        mem.add(keys, values)
        np.testing.assert_array_equal(mem.mem['x'], [[1],[1]])
        np.testing.assert_array_equal(mem.mem['a'], [[1],[1]])


        values = dict(x=np.array([[2],[3]]), a=np.array([[2],[3]]))
        mem.add(keys, values)
        np.testing.assert_array_equal(mem.mem['x'], [[2],[3]])
        np.testing.assert_array_equal(mem.mem['a'], [[2],[3]])

        values = dict(x=np.array([[4],[5],[6]]), a=np.array([[4],[5],[6]]))
        mem.add(keys, values)
        np.testing.assert_array_equal(mem.mem['x'], [[5],[6]])
        np.testing.assert_array_equal(mem.mem['a'], [[5],[6]])

        values = dict(x=np.array([[6]]), a=np.array([[6]]))
        mem.add(keys, values)
        sample = mem.sample(keys, 1)
        np.testing.assert_array_equal(sample['x'], [[6]])
        np.testing.assert_array_equal(sample['a'], [[6]])

        # large batch size
        mem = hgail.misc.utils.KeyValueReplayMemory(maxsize=200)

        paths = [1] * 50
        values = dict(x=paths, a=paths)
        mem.add(keys, values)
        sample = mem.sample(keys, 10)
        self.assertEqual(len(sample['x']), 10)
        self.assertEqual(len(sample['a']), 10)

        paths = [1] * 500
        values = dict(x=paths, a=paths)
        mem.add(keys, values)
        self.assertEqual(len(mem.mem['x']), 200)
        self.assertEqual(len(mem.mem['a']), 200)

        # realistic values
        keys = ['x','a']
        mem = hgail.misc.utils.KeyValueReplayMemory(maxsize=200)
        x = np.arange(100).reshape(25,4)
        a = np.ones((25,2))
        values = dict(x=x, a=a)
        mem.add(keys, values)
        sample = mem.sample(keys, 1)
        self.assertEqual(sample['x'].shape, (1,4))
        self.assertEqual(sample['a'].shape, (1,2))
        sample = mem.sample(keys, 2)
        self.assertEqual(sample['x'].shape, (2,4))
        self.assertEqual(sample['a'].shape, (2,2))

class TestNumpyUtils(unittest.TestCase):

    def test_softmax(self):

        logits = np.array([ 0.866008,  0.393137, -0.490158])
        probs = hgail.misc.utils.softmax(logits)

class TestActionRangeNormalizer(unittest.TestCase):

    def test_normalize(self):
        low = [-1,-2]
        high = [2,2]
        normalizer = hgail.misc.utils.ActionRangeNormalizer(low, high)
        
        inputs = [
            [-1,-2],
            [2,2],
            [.5,0]
        ]
        outputs = [
            [-1,-1],
            [1,1],
            [0,0]
        ]
        for (inpt, out) in zip(inputs, outputs):
            actual = normalizer(inpt)
            np.testing.assert_array_almost_equal(out, actual)

if __name__ == '__main__':
    unittest.main()