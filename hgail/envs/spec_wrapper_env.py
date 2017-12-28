
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv

class SpecWrapperEnv(ProxyEnv):
    def __init__(self, wrapped_env, action_space, observation_space):
        Serializable.quick_init(self, locals())
        super(SpecWrapperEnv, self).__init__(wrapped_env)
        self._action_space = action_space
        self._observation_space = observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
