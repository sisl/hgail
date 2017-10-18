
import numpy as np

from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.sampler.base import BaseSampler

class HierarchySampler(BaseSampler):
    '''
    This class extracts some latent depth from given paths and then 
    calls its super class to perform the actual sample processing
    '''
    def __init__(self, algo, depth=1, **kwargs):
        super(HierarchySampler, self).__init__(algo, **kwargs)
        assert depth > 0, 'use base sampler for depth == 0'
        self.depth = depth

    @overrides
    def process_samples(self, itr, paths):
        paths = self._extract_depth(paths)
        return super(HierarchySampler, self).process_samples(itr, paths)

    def _extract_depth(self, paths):
        # iterate paths extracting the relevant depth
        depth_paths = []
        for path in paths:

            # select the relevant info dict
            info = path['agent_infos']
            for _ in range(self.depth):
                info = info['latent_info']

            # extract latent information
            idxs = np.where(np.array(info['update']) != False)[0]
            assert len(idxs) >= 1, 'at least one action update must have been made'
            actions = info['latent'][idxs]
            probs = info['prob'][idxs]
            latent = info['latent_info']['latent'][idxs]

            # subselect observations to those in which a decision was actually made
            observations = path['observations'][idxs]
                
            # compute discounted, multi-step reward for each action
            rewards = []
            idxs = list(idxs) + [len(path['rewards'])]
            for (s,e) in zip(idxs, idxs[1:]):
                reward = special.discount_cumsum(
                    path['rewards'][s:e], self.algo.discount)[0]
                rewards.append(reward)

            # package each path back into dictionary
            depth_paths.append(dict(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    agent_infos=dict(
                        prob=probs,
                        latent=latent
                    ),
                    env_infos=dict() 
                )
            )

        return depth_paths

