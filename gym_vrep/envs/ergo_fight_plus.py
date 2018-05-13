import gym
import numpy as np
import torch
from gym_vrep.models.model_lstm_v3 import LstmNetRealv3
from torch import load
from torch.autograd import Variable


class ErgoFightPlusWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ErgoFightPlusWrapper, self).__init__(env)
        self.env = env
        self.load_model(LstmNetRealv3(nodes=128, layers=5), "../models/lstm_v2_exp6_l5_n128.pt")

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = load(modelPath, map_location="cpu")
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
        print("DBG: MODEL LOADED:", modelPath)

    @staticmethod
    def double_unsqueeze(data):
        return torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)

    @staticmethod
    def double_squeeze(data):
        return torch.squeeze(torch.squeeze(data)).data.cpu().numpy()

    def data_to_var(self, sim_t2, real_t1, action):
        return Variable(
            self.double_unsqueeze(torch.cat(
                [torch.from_numpy(sim_t2),
                 torch.from_numpy(real_t1),
                 torch.from_numpy(action)], dim=0)))

    def step(self, action):
        obs_real_t1 = self.unwrapped._self_observe()
        obs_sim_t2, rew, done, info = self.unwrapped.step(action)

        variable = self.data_to_var(obs_sim_t2[:12].copy(), obs_real_t1[:12].copy(), np.array(action).copy())

        obs_real_t2_delta = self.double_squeeze(self.net.forward(variable))

        obs_real_t2 = obs_sim_t2[:12].copy() + obs_real_t2_delta

        obs_real_t2_clip = np.clip(obs_real_t2, -1, 1)

        # DENORMALIZE, AND APPLY INTERNALLY
        obs_real_t2_denorm_pos = self.unwrapped._denormalize(obs_real_t2_clip[:6])
        obs_real_t2_denorm_vel = self.unwrapped._denormalizeVel(obs_real_t2_clip[6:])

        self.unwrapped.set_state(obs_real_t2_denorm_pos, obs_real_t2_denorm_vel)

        # print("real t1:", obs_real_t1[:12].round(2))
        # print("sim_ t2:", obs_sim_t2[:12].round(2))
        # print("action_:", action.round(2))
        # print("real t2:", obs_real_t2[:12].round(2))
        # print("===")

        return self.unwrapped._self_observe(), rew, done, info

    def reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        return self.env.reset()


def ErgoFightPlusEnv(base_env_id):
    return ErgoFightPlusWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    env = gym.make("ErgoFightStatic-Headless-Shield-Move-HalfRand-Plus-v0")

    env.reset()

    for episode in range(1):
        for step in range(5):
            action = env.action_space.sample()
            env.step(action)
