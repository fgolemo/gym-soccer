import os
import time

import gym
import numpy as np
from gym import error, spaces
import math

import logging

from gym_vrep.envs.constants import JOINT_LIMITS
from gym_vrep.envs.normalized_wrapper import NormalizedActWrapper, NormalizedObsWrapper
from vrepper.core import vrepper

logger = logging.getLogger(__name__)

JOINT_LIMITS_MAXMIN = [-150, 150]

VELOCITY_LIMITS = [-np.inf, np.inf]  # observations

REST_POS = [0, -90, 35, 0, 55, -90]

BALL_POS = [0, 0.05, .28]


class ErgoBallThrowAirtimeEnv(gym.Env):
    vrep_running = False
    max_z = 0

    def __init__(self, headless=True):
        self.headless = headless
        self._startEnv(headless)

        joint_boxes = spaces.Box(low=JOINT_LIMITS_MAXMIN[0], high=JOINT_LIMITS_MAXMIN[1], shape=6)

        obs_box = spaces.Box(low=-np.inf, high=np.inf,
                             shape=(6 + 6 + 3))  # 6 joint pos, 6 joint vel, 3 ball corrdinates

        self.observation_space = obs_box
        self.action_space = joint_boxes

        self.minima = [JOINT_LIMITS[i][0] for i in range(6)]
        self.maxima = [JOINT_LIMITS[i][1] for i in range(6)]

    def _startEnv(self, headless):
        self.venv = vrepper(headless=headless)
        self.venv.start()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.venv.load_scene(current_dir + '/../scenes/poppy_ergo_jr_vanilla_ball2.ttt')
        motors = []
        for i in range(6):
            motor = self.venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
            motors.append(motor)
        self.motors = motors
        self.ball = self.venv.get_object_by_name("static_ball")
        self.ball_collision = self.venv.get_collision_object("ballcoll")

    def _restPos(self):
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=False)

        for i, m in enumerate(self.motors):
            m.set_position_target(REST_POS[i])

        self.ball.set_position(*BALL_POS)

        if self.headless:
            time.sleep(.4)  # I have yet to test this, is faster if headless
        else:
            time.sleep(.5)

        self.venv.make_simulation_synchronous(True)

    def _reset(self):
        self._restPos()
        self._self_observe()
        return self.observation

    def _getReward(self):
        reward = 0
        if not self.ball_collision.is_colliding():
            # then the ball is currently in the air
            reward = 10

        return reward

    def _self_observe(self):
        pos = []
        forces = []
        for m in self.motors:
            pos.append(m.get_joint_angle())
            forces.append(m.get_joint_velocity()[0])

        self.observation = np.hstack((pos, forces, self.ball.get_position())).astype('float32')

    def _gotoPos(self, pos):
        for i, m in enumerate(self.motors):
            m.set_position_target(pos[i])

    def _clipActions(self, actions):
        a = []
        for i, action in enumerate(actions):
            a.append(np.clip(action, self.minima[i], self.maxima[i]))
        return np.array(a)

    def _step(self, actions):
        actions = self._clipActions(actions)

        # step
        self._gotoPos(actions)
        self.venv.step_blocking_simulation()

        # observe again
        self._self_observe()

        return self.observation, self._getReward(), False, {}

    def _close(self):
        self.venv.stop_simulation()
        self.venv.end()

    def _render(self, mode='human', close=False):
        pass


def ErgoBallThrowAirtimeNormHEnv(env_id):
    return NormalizedObsWrapper(NormalizedActWrapper(gym.make(env_id)))


def ErgoBallThrowAirtimeNormGEnv(env_id):
    return NormalizedObsWrapper(
        NormalizedActWrapper(gym.make(env_id)))


if __name__ == '__main__':
    import gym_vrep

    env = gym.make("ErgoBallThrowAirtime-Graphical-Normalized-v0")

    for k in range(3):
        observation = env.reset()
        for i in range(30):
            if i % 5 == 0:
                # action = env.action_space.sample() # this doesn't work
                action = np.random.uniform(low=-1.0, high=1.0, size=(6))
            observation, reward, done, info = env.step(action)
            print(action, observation, reward)

    env.close()

    print('simulation ended. leaving in 5 seconds...')
    time.sleep(2)
