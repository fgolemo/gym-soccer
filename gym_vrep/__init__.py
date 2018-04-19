import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ErgoBall-v0',
    entry_point='gym_vrep.envs:ErgoBallEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=False,
)

register(
    id='ErgoBallDyn-v0',
    entry_point='gym_vrep.envs:ErgoBallDynEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallDyn-v1',
    entry_point='gym_vrep.envs:ErgoBallDynRewEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallThrow-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallThrowRandom-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowRandEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallThrowVert-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowVertEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallThrowVertRand-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowVertRandEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallThrowVert-v1',
    entry_point='gym_vrep.envs:ErgoBallThrowVertMaxEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='ErgoBallThrowAirtime-Headless-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=80,
    kwargs={'headless': True},
    reward_threshold=500.0
    # nondeterministic = True,
)

register(
    id='ErgoBallThrowAirtime-Graphical-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=80,
    kwargs={'headless': False},
    reward_threshold=500.0
)

register(
    id='ErgoBallThrowAirtime-Headless-Random-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=80,
    kwargs={'headless': True, 'random': True},
    reward_threshold=500.0
)

register(
    id='ErgoBallThrowAirtime-Graphical-Random-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=80,
    kwargs={'headless': False, 'random': True},
    reward_threshold=500.0
)

register(
    id='ErgoBallThrowAirtime-Headless-Random-Height-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=80,
    kwargs={'headless': True, 'random': True, 'height_based_reward': True},
    reward_threshold=500.0
)

register(
    id='ErgoBallThrowAirtime-Graphical-Random-Height-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeEnv',
    timestep_limit=80,
    kwargs={'headless': False, 'random': True, 'height_based_reward': True},
    reward_threshold=500.0
)

register(
    id='ErgoBallThrowAirtime-Headless-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormHEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Headless-v0'}
)

register(
    id='ErgoBallThrowAirtime-Graphical-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormGEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Graphical-v0'}
)

register(
    id='ErgoBallThrowAirtime-Headless-Random-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormHEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Headless-Random-v0'}
)

register(
    id='ErgoBallThrowAirtime-Graphical-Random-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormGEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Graphical-Random-v0'}
)

register(
    id='ErgoBallThrowAirtime-Headless-Random-Height-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormHEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Headless-Random-Height-v0'}
)

register(
    id='ErgoBallThrowAirtime-Graphical-Random-Height-Normalized-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowAirtimeNormGEnv',
    kwargs={'env_id': 'ErgoBallThrowAirtime-Graphical-Random-Height-v0'}
)

register(
    id='ErgoFightStatic-Graphical-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False},
)

register(
    id='ErgoFightStatic-Headless-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True},
)

register(
    id='ErgoFightStatic-Graphical-NoImg-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'with_img': False},
)

register(
    id='ErgoFightStatic-Headless-NoImg-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'with_img': False},
)

register(
    id='ErgoFightStatic-Graphical-OnlyImg-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'only_img': True},
)

register(
    id='ErgoFightStatic-Headless-OnlyImg-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'only_img': True},
)

register(
    id='ErgoFightStatic-Graphical-Fencing-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'fencing_mode': True, 'with_img': False},
)

register(
    id='ErgoFightStatic-Headless-Fencing-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'fencing_mode': True, 'with_img': False},
)

register(
    id='ErgoFightStatic-Graphical-Fencing-Swordonly-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'fencing_mode': True, 'with_img': False, 'sword_only': True},
)

register(
    id='ErgoFightStatic-Headless-Fencing-Swordonly-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'fencing_mode': True, 'with_img': False, 'sword_only': True},
)

register(
    id='ErgoFightStatic-Graphical-Fencing-Swordonly-Fat-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'fencing_mode': True, 'with_img': False, 'sword_only': True, 'fat': True},
)

register(
    id='ErgoFightStatic-Headless-Fencing-Swordonly-Fat-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'fencing_mode': True, 'with_img': False, 'sword_only': True, 'fat': True},
)

register(
    id='ErgoFightStatic-Graphical-Fencing-Swordonly-Fat-NoMove-HalfRand-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'fencing_mode': True, 'with_img': False, 'sword_only': True, 'fat': True,
            'no_move': True, 'scaling': 0.5},
)

register(
    id='ErgoFightStatic-Headless-Fencing-Swordonly-Fat-NoMove-HalfRand-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'fencing_mode': True, 'with_img': False, 'sword_only': True, 'fat': True,
            'no_move': True, 'scaling': 0.5},
)

register(
    id='ErgoFightStatic-Graphical-Fencing-Defence-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': False, 'fencing_mode': True, 'with_img': False, 'defence': True},
)

register(
    id='ErgoFightStatic-Headless-Fencing-Defence-v0',
    entry_point='gym_vrep.envs:ErgoFightStaticEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'headless': True, 'fencing_mode': True, 'with_img': False, 'defence': True},
)
