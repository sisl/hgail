
import copy
import numpy as np
import unittest

import hgail.misc.datasets

class TestCriticDataset(unittest.TestCase):

    def test_batches(self):
        x = np.array([[1,2],[3,4],[5,6]])
        a = np.array([[1,2],[3,4],[5,6]])
        data = {'observations': x, 'actions': a}
        dataset = hgail.misc.datasets.CriticDataset(data, batch_size=1, shuffle=False)
        samples_data = {'observations': x + .5, 'actions': a + .5}
        itr = dataset.batches(samples_data)

        for (i, batch) in enumerate(dataset.batches(samples_data)):
            np.testing.assert_array_equal(batch['rx'], [x[i]])
            np.testing.assert_array_equal(batch['ra'], [a[i]])
            np.testing.assert_array_equal(batch['gx'], [x[i] + .5])
            np.testing.assert_array_equal(batch['ga'], [a[i] + .5])

        samples_data = {
            'observations': np.array([[0,0]]), 
            'actions': np.array([[-1,-1]])
        }

        itr = dataset.batches(samples_data)
        batches = list(itr)
        self.assertTrue(len(batches) == 1)
        np.testing.assert_array_equal(batches[0]['rx'], [x[0]])
        np.testing.assert_array_equal(batches[0]['ra'], [a[0]])
        np.testing.assert_array_equal(batches[0]['gx'], [[0,0]])
        np.testing.assert_array_equal(batches[0]['ga'], [[-1,-1]])

        # since the previous samples_data was len 1, the current 
        # index is 1 in the dataset
        samples_data = {
            'observations': np.array([[0,0],[0,0],[0,0],[10,10]]), 
            'actions': np.array([[-1,-1],[-1,-1],[-1,-1],[-11,-11]])
        }
        itr = dataset.batches(samples_data)
        batches = list(itr)
        self.assertTrue(len(batches) == 4)
        np.testing.assert_array_equal(batches[0]['rx'], [x[1]])
        np.testing.assert_array_equal(batches[0]['ra'], [a[1]])
        np.testing.assert_array_equal(batches[0]['gx'], [[0,0]])
        np.testing.assert_array_equal(batches[0]['ga'], [[-1,-1]])
        np.testing.assert_array_equal(batches[3]['rx'], [x[1]])
        np.testing.assert_array_equal(batches[3]['ra'], [a[1]])
        np.testing.assert_array_equal(batches[3]['gx'], [[10,10]])
        np.testing.assert_array_equal(batches[3]['ga'], [[-11,-11]])

        # test shuffling
        np.random.seed(6)
        data = {'observations': np.array([[0],[1]]), 'actions': np.array([[2],[3]])}
        dataset = hgail.misc.datasets.CriticDataset(data, batch_size=1, shuffle=True)
        samples_data = {
            'observations': np.array([[4],[5]]),
            'actions': np.array([[6],[7]])
        }
        # go through once before shuffling
        itr = dataset.batches(samples_data)
        list(itr)
        itr = dataset.batches(samples_data)
        batch = next(itr)
        np.testing.assert_array_equal(batch['rx'], [[1]])
        np.testing.assert_array_equal(batch['ra'], [[3]])

    def test_batches_uneven(self):
        x = np.array([[1,2],[3,4],[5,6]])
        a = np.array([[1,2],[3,4],[5,6]])
        data = {'observations': x, 'actions': a}
        dataset = hgail.misc.datasets.CriticDataset(data, batch_size=5, shuffle=False)
        samples_data = {
            'observations': np.array([[1,2]]), 
            'actions': np.array([[1,2]])
        }
        batches = list(dataset.batches(samples_data))
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]['rx']), 5)
        self.assertEqual(len(batches[0]['ra']), 5)
        self.assertEqual(len(batches[0]['ga']), 5)
        self.assertEqual(len(batches[0]['gx']), 5)

        samples_data = {
            'observations': np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]]),
            'actions': np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]])
        }
        batches = list(dataset.batches(samples_data))
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]['rx']), 5)
        self.assertEqual(len(batches[0]['ra']), 5)
        self.assertEqual(len(batches[0]['ga']), 5)
        self.assertEqual(len(batches[0]['gx']), 5)
        self.assertEqual(len(batches[1]['rx']), 5)
        self.assertEqual(len(batches[1]['ra']), 5)
        self.assertEqual(len(batches[1]['ga']), 5)
        self.assertEqual(len(batches[1]['gx']), 5)

class TestRecognitionDataset(unittest.TestCase):

    def test_batches(self):
        x = np.array([[1,2],[3,4],[5,6]])
        a = np.array([[1,2],[3,4],[5,6]])
        c = np.array([[1,0],[0,1],[1,0]])
        samples_data = {'observations': x, 'actions': a, 'agent_infos':{'latent':c}}

        # batch size 1
        dataset = hgail.misc.datasets.RecognitionDataset(batch_size=1)
        batches = list(dataset.batches(samples_data))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0]['x'].shape, (1,2))
        self.assertEqual(batches[0]['a'].shape, (1,2))
        self.assertEqual(batches[0]['c'].shape, (1,2))

        # batch size < size of data and uneven
        dataset = hgail.misc.datasets.RecognitionDataset(batch_size=2)
        batches = list(dataset.batches(samples_data))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[-1]['x'].shape, (2,2))
        self.assertEqual(batches[-1]['a'].shape, (2,2))
        self.assertEqual(batches[-1]['c'].shape, (2,2))

        # batch size = size of data
        dataset = hgail.misc.datasets.RecognitionDataset(batch_size=3)
        batches = list(dataset.batches(samples_data))
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0]['x'], np.array([[1,2],[3,4],[5,6]]))
        np.testing.assert_array_equal(batches[0]['a'], np.array([[1,2],[3,4],[5,6]]))
        np.testing.assert_array_equal(batches[0]['c'], np.array([[1,0],[0,1],[1,0]]))
        
        # batch size > size of data
        dataset = hgail.misc.datasets.RecognitionDataset(batch_size=4)
        batches = list(dataset.batches(samples_data))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[-1]['x'].shape, (4,2))
        self.assertEqual(batches[-1]['a'].shape, (4,2))
        self.assertEqual(batches[-1]['c'].shape, (4,2))

    def test_batches_recurrent_conditional(self):
        x = np.zeros((10,5,3))
        a = np.ones((10,5,3))
        c = np.ones((10,5,3)) * 2
        z = np.ones((10,5,3)) * 3
        samples_data = {
            'observations': x, 
            'actions': a, 
            'agent_infos': {
                'latent':c, 
                'z':z},
            'valids': np.ones((10,5))
        }
        dataset = hgail.misc.datasets.RecognitionDataset(
            batch_size=3,
            conditional=True,
            cond_key='z',
            recurrent=True
        )
        batches = list(dataset.batches(samples_data))
        
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[-1]['x'].shape, (3,5,6))
        self.assertEqual(batches[-1]['a'].shape, (3,5,3))
        self.assertEqual(batches[-1]['c'].shape, (3,5,3))
        self.assertEqual(batches[-1]['valids'].shape, (3,5))

if __name__ == '__main__':
    unittest.main()