#from AbstractAgent import AbstractAgent

from gym import spaces
import numpy as np

from model import DQN
from replay_buffer import ReplayBuffer
import gym
#device = "cuda"

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
#import pickle


class MyAgent():
    def __init__(self, observation_space, action_space):
        shape = observation_space.shape
        print("!~!@ in agnet env obs shape", observation_space.shape)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)
        print("!~!@ in agent env obs shape", self.observation_space.shape)
    
        #self.observation_space = observation_space
        self.action_space = action_space

        self.device = 'cpu'
        self.policy_network = DQN(self.observation_space,
                         self.action_space   )
        #print('net shape', self.policy_network.shape)
        self.policy_network.load_state_dict(torch.load(
            './policy_network_3000000', map_location=torch.device(self.device)))

        self.policy_network.eval()
        # TODO Initialise your agent's models

        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        action_list = [1,2,18,6,12,19,20] 
  
        state = np.rollaxis(state, 2)
        
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            #print("~~~~~~~~", self.action_space.n)
            #print('asasdsad', action.item())
            if action not in action_list:
                return 0

            return action.item()

  
