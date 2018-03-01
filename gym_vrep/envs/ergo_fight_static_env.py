import os
import time
import gym
import numpy as np
from gym import spaces
import logging

from pytorch_a2c_ppo_acktr.inference import Inference
from skimage.transform import resize
from gym_vrep.envs.constants import JOINT_LIMITS, BALL_STATES, RANDOM_NOISE
from gym_vrep.envs.normalized_wrapper import NormalizedActWrapper, NormalizedObsWrapper
from vrepper.core import vrepper

logger = logging.getLogger(__name__)

REST_POS = [0, 0, 0, 0, 0, 0]
RANDOM_NOISE = [
    (-90, 90),
    (-30, 30),
    (-30, 30),
    (-45, 45),
    (-30, 30),
    (-30, 30)
]
INVULNERABILITY_AFTER_HIT = 3  # how many frames after a hit to reset
IMAGE_SIZE = (84, 84)


class ErgoFightStaticEnv(gym.Env):
    def __init__(self, headless=True, with_img=True,
                 only_img=False, fencing_mode=False, defence=False):
        self.headless = headless
        self.with_img = with_img
        self.only_img = only_img
        self.fencing_mode = fencing_mode
        self.defence = defence

        if self.defence:
            # load up the inference model for the attacker
            self.inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                                 "trained_models/ppo/"
                                 "ErgoFightStatic-Headless-Fencing-v0-180209225957.pt")

        self._startEnv(headless)

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        joint_boxes = spaces.Box(low=-1, high=1, shape=6)

        if self.with_img:
            cam_image = spaces.Box(low=0, high=255, shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

            if self.only_img:
                self.observation_space = cam_image
            else:
                own_joints = spaces.Box(low=-1, high=1, shape=(6 + 6))  # 6 joint pos, 6 joint vel
                self.observation_space = spaces.Tuple((cam_image, own_joints))
        else:
            # 6 own joint pos, 6 own joint vel, 6 enemy joint pos, 6 enemy joint vel
            all_joints = spaces.Box(low=-1, high=1, shape=(6 + 6 + 6 + 6))
            self.observation_space = all_joints

        self.action_space = joint_boxes

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]
        self.frames_after_hit = -1  # -1 means no recent hit, anything 0 or above means it's counting

    def _seed(self, seed=None):
        np.random.seed(seed)

    def _startEnv(self, headless):
        self.venv = vrepper(headless=headless)
        self.venv.start()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.venv.load_scene(current_dir + '/../scenes/poppy_ergo_jr_fight_sword1.ttt')
        self.motors = ([], [])
        for robot_idx in range(2):
            for motor_idx in range(6):
                motor = self.venv.get_object_by_name('r{}m{}'.format(robot_idx + 1, motor_idx + 1), is_joint=True)
                self.motors[robot_idx].append(motor)
        self.sword_collision = self.venv.get_collision_object("sword_hit")
        self.cam = self.venv.get_object_by_name('cam', is_joint=False).handle
        # self.tip = self.frames_after_hit

    def _restPos(self):
        self.done = False
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=True)

        for i, m in enumerate(self.motors[0]):
            m.set_position_target(REST_POS[i])

        self.randomize(robot=1)

        for _ in range(15):  # TODO test if 15 frames is enough
            self.venv.step_blocking_simulation()

    def randomize(self, robot=1):
        for i in range(6):
            new_pos = REST_POS[i] + np.random.randint(
                low=RANDOM_NOISE[i][0],
                high=RANDOM_NOISE[i][1],
                size=1)[0]
            self.motors[robot][i].set_position_target(new_pos)

    def _reset(self):
        self._restPos()
        self._self_observe()
        self.frames_after_hit = -1  # this enables hits / disables invulnerability frame
        return self.observation

    def _getReward(self):
        # The only way of getting reward is by hitting and releasing, hitting and releasing.
        # Just touching and holding doesn't work.
        reward = 0
        if self.sword_collision.is_colliding() and self.frames_after_hit == -1:
            reward = 1
            if not self.fencing_mode:
                self.frames_after_hit = 0
            else:
                self._restPos()  # if fencing mode then reset pos on each hit

        # the following bit is for making sure the robot doen't just hit repeatedly
        # ...so the invulnerability countdown only start when the collision is released
        else:  # if it's not hitting anything right now
            if self.frames_after_hit >= 0:
                self.frames_after_hit += 1
            if self.frames_after_hit >= INVULNERABILITY_AFTER_HIT:
                self.frames_after_hit = -1

        if self.defence:
            reward *= -1

        return reward

    def _get_robot_posvel(self, robot_id):
        pos = []
        vel = []
        for i, m in enumerate(self.motors[robot_id]):
            pos.append(m.get_joint_angle())
            vel.append(m.get_joint_velocity()[0])

        pos = self._normalize(pos)  # move pos into range [-1,1]

        joint_vel = np.hstack((pos, vel)).astype('float32')
        return joint_vel

    def _self_observe(self):
        own_joint_vel = self._get_robot_posvel(0)
        if self.with_img:
            cam_image = self.venv.flip180(self.venv.get_image(self.cam))
            cam_image = resize(cam_image, IMAGE_SIZE)
            if self.only_img:
                self.observation = cam_image
            else:
                self.observation = (cam_image, own_joint_vel)
        else:
            enemy_joint_vel = self._get_robot_posvel(1)
            self.observation = np.hstack((own_joint_vel, enemy_joint_vel)).astype('float32')

    def _gotoPos(self, pos, robot=0):
        for i, m in enumerate(self.motors[robot]):
            m.set_position_target(pos[i])

    def _normalize(self, pos):
        out = []
        for i in range(6):
            shifted = (pos[i] - JOINT_LIMITS[i][0]) / self.diffs[i]  # now it's in [0,1]
            norm = shifted * 2 - 1
            out.append(norm)
        return out

    def _denormalize(self, actions):
        out = []
        for i in range(6):
            shifted = (actions[i] + 1) / 2  # now it's within [0,1]
            denorm = shifted * self.diffs[i] + JOINT_LIMITS[i][0]
            out.append(denorm)
        return out

    def _step(self, actions):
        actions = np.clip(actions, -1, 1)  # first make sure actions are normalized
        actions = self._denormalize(actions)  # then scale them to the actual joint angles
        robot = 0

        # step
        if self.defence:
            attacker_action = self.inf.get_action(self.observation)
            self._gotoPos(attacker_action, robot=0)
            robot = 1

            #TODO: why is this less aggressive than the direct inference???


        self._gotoPos(actions, robot=robot)
        self.venv.step_blocking_simulation()

        # observe again
        self._self_observe()

        return self.observation, self._getReward(), self.done, {}

    def _close(self):
        self.venv.stop_simulation()
        self.venv.end()

    def _render(self, mode='human', close=False):
        # This intentionally does nothing and is only here for wrapper functions.
        # if you want graphical output, use the environments
        # "ErgoBallThrowAirtime-Graphical-Normalized-v0"
        # or
        # "ErgoBallThrowAirtime-Graphical-v0"
        # ... not the ones with "...-Headless-..."
        pass


if __name__ == '__main__':
    import gym_vrep
    import matplotlib.pyplot as plt

    def test_normal_mode():

        env = gym.make("ErgoFightStatic-Graphical-v0")

        plt.ion()
        img = np.random.uniform(0, 255, (256, 256, 3))
        plt_img = plt.imshow(img, interpolation='none', animated=True, label="blah")
        plt_ax = plt.gca()

        for k in range(3):
            observation = env.reset()
            print("init done")
            time.sleep(2)
            for i in range(30):
                if i % 5 == 0:
                    # action = env.action_space.sample() # this doesn't work
                    action = np.random.uniform(low=-1.0, high=1.0, size=(6))
                observation, reward, done, info = env.step(action)
                plt_img.set_data(observation[0])
                plt_ax.plot([0])
                plt.pause(0.001)  # I found this necessary - otherwise no visible img
                print(action, observation[0].shape, observation[1], reward, done)
                print(".")

        env.close()

        print('simulation ended. leaving in 5 seconds...')
        time.sleep(2)

    def test_fencing_defence():
        env = gym.make("ErgoFightStatic-Graphical-Fencing-Defense-v0")

        env.reset()

        for _ in range(50):
            act = env.action_space.sample()
            obs, rew, _, _ = env.step(act)
            print (act, obs, rew)

    test_fencing_defence()
