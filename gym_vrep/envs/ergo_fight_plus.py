import gym
import numpy as np
from gym_vrep.models.model_lstm import ModelLstm
from torch import from_numpy, load
from torch.autograd import Variable

class ErgoFightPlusWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ErgoFightPlusWrapper, self).__init__(env)
        self.env = env
        self.load_model(ModelLstm(), "../models/lstm_v4_5l_256n.pt")

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location="cpu")
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
        print("DBG: MODEL LOADED:", modelPath)

    def _normalize(self, state):
        state[:3] -= PUSHER3DOF_POS_MIN  # add the minimum
        state[:3] /= PUSHER3DOF_POS_DIFF  # divide by range to bring into [0,1]

        state[3:] -= PUSHER3DOF_VEL_MIN
        state[3:] /= PUSHER3DOF_VEL_DIFF

        state *= 2  # double and
        state -= 1  # shift left by one to bring into range [-1,1]

        return state

    def _denormalize(self, state):
        state += 1
        state /= 2 # now it's back in range [0,1]

        state[:3] *= PUSHER3DOF_POS_DIFF
        state[:3] += PUSHER3DOF_POS_MIN # now it's uncentered and shifted

        state[3:] *= PUSHER3DOF_VEL_DIFF
        state[3:] += PUSHER3DOF_VEL_MIN

        return state


    def _create_input(self, obs_next, action, obs_current):
        _input = np.hstack([
            self._normalize(obs_next[:6]),
            self._normalize(obs_current[:6]),
            action
        ])
        return Variable(from_numpy(_input).float().unsqueeze(0).unsqueeze(0), volatile=True)

    def _step(self, action):
        obs_current = self.env.env._get_obs()
        obs_next, rew, done, info = self.env.step(action)
        variable = self._create_input(obs_next.copy(), action.copy(), obs_current.copy())
        obs_correction = self.net.forward(variable).data.cpu().squeeze(0).squeeze(0).numpy()
        # print(np.around(obs_next, 2), np.around(obs_correction.data.cpu().numpy(), 2))

        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()

        # FIRST NORMALIZE SIMULATED RESULTS,
        normalized_obs = self._normalize(obs_next[:6].copy())
        # THEN APPLY CORRECTION,
        corrected_obs = normalized_obs + obs_correction
        # DENORMALIZE, AND APPLY INTERNALLY
        denormalized_obs = self._denormalize(corrected_obs)

        qpos = np.hstack((denormalized_obs[:3], qpos[3:]))
        qvel = np.hstack((denormalized_obs[3:], qvel[3:]))

        self.env.env.set_state(qpos, qvel)

        # Update the environnement with the new state
        return self.env.env._get_obs(), rew, done, info

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        return self.env.reset()


def ErgoFightPlusEnv(base_env_id):
    return ErgoFightPlusWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    env = gym.make("ErgoFightStatic-Graphical-Shield-Move-HalfRand-Plus-v0")