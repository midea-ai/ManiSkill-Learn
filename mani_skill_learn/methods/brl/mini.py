"""
Behavior cloning(BC)
"""
import torch
import torch.nn.functional as F

from mani_skill_learn.networks import build_model, hard_update, soft_update, build_backbone
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape, combine_obs_with_action
from ..builder import BRL


@BRL.register_module()
class Mini(BaseAgent):
    def __init__(self, backbone_cfg, policy_cfg, value_cfg, obs_shape, action_shape, action_space, batch_size=128, gamma=0.99, 
    update_coeff=0.005, action_noise=0.2, noise_clip=0.5, policy_update_interval=2, alpha=2.5, reward_scale=1):
        super(Mini, self).__init__()
        backbone_optim_cfg = backbone_cfg.pop("optim_cfg")
        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        value_optim_cfg = value_cfg.pop("optim_cfg")
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_coeff = update_coeff
        self.policy_update_interval = policy_update_interval
        self.action_noise = action_noise
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.lstm_len = 1

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        value_cfg['obs_shape'] = obs_shape
        value_cfg['action_shape'] = action_shape

        # backbone_cfg['obs_shape'] = obs_shape
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        backbone_cfg = replace_placeholder_with_args(backbone_cfg, **replaceable_kwargs)
        self.backbone = build_backbone(backbone_cfg)
        self.policy = build_model(policy_cfg)
        self.critic = build_model(value_cfg)

        self.target_policy = build_model(policy_cfg)
        self.target_critic = build_model(value_cfg)

        hard_update(self.target_critic, self.critic)
        hard_update(self.target_policy, self.policy)

        self.backbone_optim = build_optimizer(self.backbone, backbone_optim_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)

    def forward(self, obs, **kwargs):
        from mani_skill_learn.utils.data import to_torch
        obs = to_torch(obs, dtype='float32', device=self.device)
        obs_z = self.backbone(obs)
        return self.policy(obs_z, **kwargs)

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        sampled_batch['rewards'] = self.reward_scale * sampled_batch['rewards']

        with torch.no_grad():
            sampled_next_z = self.backbone(sampled_batch['next_obs'])
            # _, _, next_mean_action, _, _ = self.target_policy(sampled_batch['next_obs'], mode='all')
            _, _, next_mean_action, _, _ = self.target_policy(sampled_next_z, mode='all')
            noise = (torch.randn_like(next_mean_action) * self.action_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.target_policy['policy_head'].clamp_action(next_mean_action + noise)
            # q_next_target = self.target_critic(sampled_batch['next_obs'], next_action)
            q_next_target = self.target_critic(sampled_next_z, next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * min_q_next_target


        sampled_z = self.backbone(sampled_batch['obs'])
        # q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        q = self.critic(sampled_z, sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        if updates % self.policy_update_interval == 0:
            # pred_action = self.policy(sampled_batch['obs'], mode='eval')
            pred_action = self.policy(sampled_z, mode='eval')
            # q = self.critic(sampled_batch['obs'], pred_action)[..., 0]
            q = self.critic(sampled_z, pred_action)[..., 0]
            lmbda = self.alpha / (q.abs().mean().detach() + 1E-5)
            bc_loss = F.mse_loss(pred_action, sampled_batch['actions'])
            policy_loss = -lmbda * q.mean() + bc_loss
            bc_abs_error = torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean()

            self.backbone_optim.zero_grad()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.backbone_optim.step()
            self.policy_optim.step()

            soft_update(self.target_critic, self.critic, self.update_coeff)
            soft_update(self.target_policy, self.policy, self.update_coeff)
        else:
            policy_loss = torch.zeros(1)
            bc_loss = torch.zeros(1)
            lmbda = torch.zeros(1)
            bc_abs_error = torch.zeros(1)

        return {
            'critic_loss': critic_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'policy_loss': policy_loss.item(),
            'bc_loss': bc_loss.item(),
            'bc_abs_error': bc_abs_error.item(),
            'lmbda': lmbda.item(),
        }
