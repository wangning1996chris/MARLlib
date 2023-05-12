from pettingzoo.utils.conversions import parallel_wrapper_fn
from .utils.simple_env import SimpleEnv, make_env
from .mapd import Scenario
import os


class raw_env(SimpleEnv):
    def __init__(self, max_cycles=50, continuous_actions=True):
        scenario = Scenario()
        world = scenario.make_world(N=3)
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "mapd_3_agent_v1"


# Mapd_env = raw_env()
# Mapd_env.reset()
# Mapd_env.render(mode='human')
# os.system("pause")

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)