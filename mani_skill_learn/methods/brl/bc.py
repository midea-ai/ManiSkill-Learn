"""
Behavior cloning(BC)
"""
import torch
import torch.nn.functional as F

from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.torch import BaseAgent
from ..builder import BRL


@BRL.register_module()
class BC(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128, lstm_len=1):
        super(BC, self).__init__()
        self.batch_size = batch_size

        policy_optim_cfg = policy_cfg.pop("optim_cfg")

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.policy = build_model(policy_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)

        self.lstm_len = lstm_len

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size, seq_length=self.lstm_len)
        # print("before dict: batch cnt: %s", str(sampled_batch['cnt'][:15]))
        # for key, value in sampled_batch.items():
        #     print("{0} = {1}".format(key, len(value)))
        sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"], cnt=sampled_batch['cnt'])
        # sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"])
        # print("after dict: batch cnt: %s", str(sampled_batch['cnt'][:15]))
        sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        pred_action = self.policy(sampled_batch['obs'], mode='eval')
        true_action = sampled_batch['actions'][self.lstm_len-1::self.lstm_len]
        # true_idx = sampled_batch['cnt'][self.lstm_len-1::self.lstm_len]
        # print('true_action', true_action.shape)
        # print('true_idx : %s' % str(true_idx[:15]))
        # print('pred_action', pred_action.shape)
        # policy_loss = F.mse_loss(pred_action, sampled_batch['actions'])
        policy_loss = F.mse_loss(pred_action, true_action)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # return {
        #     'policy_abs_error': torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean().item(),
        #     'policy_loss': policy_loss.item()
        # }
        return {
            'policy_abs_error': torch.abs(pred_action - true_action).sum(-1).mean().item(),
            'policy_loss': policy_loss.item()
        }
