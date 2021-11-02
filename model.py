from gym import spaces
import torch.nn.functional as F
import torch.nn as nn


# Class inheritance example with Pytorch, can use Tensorflow instead.
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(action_space) == spaces.Discrete, 'action_space must be of type Discrete'

        # TODO Implement CNN layers
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 7, 512)
        self.linear2 = nn.Linear(512, action_space.n)


    def forward(self, x):
        # TODO Implement forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)  # flatten
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
