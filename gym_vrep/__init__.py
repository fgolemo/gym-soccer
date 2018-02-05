import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ErgoBall-v0',
    entry_point='gym_vrep.envs:ErgoBallEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = False,
)

register(
    id='ErgoBallDyn-v0',
    entry_point='gym_vrep.envs:ErgoBallDynEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallDyn-v1',
    entry_point='gym_vrep.envs:ErgoBallDynRewEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrow-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowRandom-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowRandEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowVert-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowVertEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowVertRand-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowVertRandEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowVert-v1',
    entry_point='gym_vrep.envs:ErgoBallThrowVertMaxEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowAirtime-Headless-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=50,
    kwargs={'headless' : True},
    reward_threshold=500.0
    # nondeterministic = True,
)

register(
    id='ErgoBallThrowAirtime-Graphical-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=50,
    kwargs={'headless' : False},
    reward_threshold=500.0
    # nondeterministic = True,
)


register(
    id='ErgoBallThrowAirtime-Headless-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormHEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Headless-v0'}
    # nondeterministic = True,
)

register(
    id='ErgoBallThrowAirtime-Graphical-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormGEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Graphical-v0'}
    # nondeterministic = True,
)