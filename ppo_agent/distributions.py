import torch
import torch.nn as nn
import torch.distributions
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from ppo_agent.utils import AddBias, init, init_normc_
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

# import torchsnooper

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1, keepdim=True)

FixedNormal.mode = lambda self: self.mean

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class DiagGaussian(nn.Module):
    def __init__(self):
        super(DiagGaussian, self).__init__()
        self.m = None

    def forward(self, action_mean, action_logstd):
        action_mean = torch.tanh(action_mean)
        zeros = torch.zeros(action_mean.size()).to(action_mean.device)

        action_logstd = action_logstd(zeros)
        # todo: change
        action_logstd = torch.sigmoid(action_logstd.exp())
        # action_logstd = torch.tanh(action_logstd)  # TODO new
        self.m = FixedNormal(action_mean, action_logstd)

    def sample(self):
        return self.m.sample()

    def mode(self):
        return self.m.mode()

    def log_probs(self, action):
        return self.m.log_probs(action)

    def entropy(self):
        return self.m.entropy()
