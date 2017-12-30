
import numpy as np

from rllab.algos import util
from rllab.misc import special, tensor_utils
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab.sampler.base import BaseSampler

import hgail.misc.utils

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
        samples_data = self._process_samples(itr, paths)
        return samples_data

    def _process_samples(self, itr, paths):
        baselines = []
        returns = []

        # compute path baselines
        all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        # compute advantages and returns at every timestep
        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        # formulate samples data, subselecting timesteps during which an action 
        # was actually taken by the policy
        observations = tensor_utils.concat_tensor_list(
            [path["observations"][path['update_idxs']] for path in paths])
        actions = tensor_utils.concat_tensor_list(
            [path["actions"][path['update_idxs']] for path in paths])
        rewards = tensor_utils.concat_tensor_list(
            [path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list(
            [path["returns"][path['update_idxs']] for path in paths])
        advantages = tensor_utils.concat_tensor_list(
            [path["advantages"][path['update_idxs']] for path in paths])

        idxs = [path['update_idxs'] for path in paths]
        hgail.misc.utils.subselect_dict_list_idxs(paths, 'env_infos', idxs)
        hgail.misc.utils.subselect_dict_list_idxs(paths, 'agent_infos', idxs)

        env_infos = tensor_utils.concat_tensor_dict_list(
            [path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list(
            [path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            advantages = util.center_advantages(advantages)

        if self.algo.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )

        logger.log("fitting baseline...")
        self.algo.baseline.fit(paths)
        logger.log("fitted")

        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        average_discounted_return = np.mean([path["returns"][0] for path in paths])
        # bug with computing entropy
        # ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))
        ent = 0.

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

    def _extract_depth(self, paths):
        # iterate paths extracting the relevant depth
        depth_paths = []
        for path in paths:

            # select the relevant info dict
            info = path['agent_infos']
            for _ in range(self.depth):
                info = info['latent_info']

            # determine the indices at which an action was actually taken
            idxs = np.where(np.array(info['update']) != False)[0]
            assert len(idxs) >= 1, 'at least one action update must have been made'

            # package each path back into dictionary
            depth_paths.append(dict(
                    observations=path['observations'],
                    actions=info['latent'],
                    rewards=path['rewards'],
                    update_idxs=idxs,
                    agent_infos=dict(
                        prob=info['prob'],
                        latent=info['latent_info']['latent']
                    ),
                    env_infos=dict() 
                )
            )

        return depth_paths
