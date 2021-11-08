import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, mini_batch_num, obs_dim, action_dim, env_key, use_gae, gamma, gae_param, num_process):
        self.env_key = env_key
        self.mini_batch_num = mini_batch_num
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rgb = torch.zeros(num_steps + 1, num_process, *obs_dim[self.env_key]['rgb'])
        self.seg = torch.zeros(num_steps + 1, num_process, *obs_dim[self.env_key]['seg'])
        self.xyz = torch.zeros(num_steps + 1, num_process, *obs_dim[self.env_key]['xyz'])
        self.agent_state = torch.zeros(num_steps + 1, num_process, obs_dim['state'])
        self.action = torch.zeros(num_steps + 1, num_process, action_dim)
        self.expert_action = torch.zeros(num_steps + 1, num_process, action_dim)
        self.rewards = torch.zeros(num_steps + 1, num_process, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_process, 1)
        self.returns = torch.zeros(num_steps + 1, num_process, 1)
        self.action_log_probs = torch.zeros(num_steps + 1, num_process, 1)
        self.masks = torch.zeros(num_steps + 1, num_process, 1)
        self.is_demo = torch.zeros(num_steps + 1, num_process, 1)
        self.num_steps = num_steps
        self.num_process = num_process
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = gae_param
        self.step = 0

    def to(self, device):
        self.rgb = self.rgb.to(device)
        self.seg = self.seg.to(device)
        self.xyz = self.xyz.to(device)
        self.agent_state = self.agent_state.to(device)
        self.action = self.action.to(device)
        self.expert_action = self.expert_action.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.masks = self.masks.to(device)
        self.is_demo = self.is_demo.to(device)
        self.rollout_device = device

    def insert(self, next_obs, action, value_preds, action_log_probs, masks, rewards, expert_action, is_demo=None):
        rgb = next_obs[self.env_key]['rgb']
        xyz = next_obs[self.env_key]['xyz']
        seg = next_obs[self.env_key]['seg']
        agent_state = next_obs['state']
        self.rgb[self.step + 1].copy_(rgb)
        self.xyz[self.step + 1].copy_(xyz)
        self.seg[self.step + 1].copy_(seg)
        self.agent_state[self.step + 1].copy_(agent_state)
        self.action[self.step].copy_(action)
        if is_demo:
            self.is_demo[self.step].copy_(is_demo)
        if expert_action:
            self.expert_action[self.step].copy_(expert_action)
        self.value_preds[self.step].copy_(value_preds)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.masks[self.step].copy_(masks)
        self.rewards[self.step].copy_(torch.tensor(rewards))
        self.step = self.step + 1
        self.step = self.step % self.num_steps

    def after_update(self):
        self.step = 0
        self.rgb[0].copy_(self.rgb[-1])
        self.xyz[0].copy_(self.xyz[-1])
        self.seg[0].copy_(self.seg[-1])
        self.agent_state[0].copy_(self.agent_state[-1])

    def get_obs(self):
        rgb = self.rgb[self.step].clone()
        xyz = self.xyz[self.step].clone()
        seg = self.seg[self.step].clone()
        agent_state = self.agent_state[self.step].clone()
        state = {self.env_key: {'rgb': rgb, 'xyz': xyz, 'seg': seg}, 'state': agent_state}
        return state

    def update_obs(self, cur_obs):
        rgb = cur_obs[self.env_key]['rgb']
        xyz = cur_obs[self.env_key]['xyz']
        seg = cur_obs[self.env_key]['seg']
        agent_state = cur_obs['state']
        self.rgb[self.step].copy_(rgb)
        self.xyz[self.step].copy_(xyz)
        self.seg[self.step].copy_(seg)
        self.agent_state[self.step].copy_(agent_state)

    def compute_returns(self, next_value):
        if self.use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step] - \
                        self.value_preds[step]
                gae = delta + self.gamma * self.tau * self.masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            length = self.rewards.size(0)
            for step in range(length):
                gae = 0
                for index in reversed(range(self.step, min(self.step + 10, length))):
                    delta = self.rewards[index] + self.gamma * self.value_preds[index + 1] * self.masks[index] - \
                            self.value_preds[index]
                    gae = delta + self.gamma * self.tau * self.masks[index] * gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages):
        batch_size = self.num_steps * self.num_process
        mini_batch_size = batch_size // self.mini_batch_num
        sampler = BatchSampler(SubsetRandomSampler(range(0, batch_size)), mini_batch_size, drop_last=False)
        data_batch = []
        for indices in sampler:
            adv = advantages.view(-1, 1)[indices]
            rgb_batch = self.rgb.view(-1, *self.obs_dim[self.env_key]['rgb'])[indices]
            xyz_batch = self.xyz.view(-1, *self.obs_dim[self.env_key]['xyz'])[indices]
            seg_batch = self.seg.view(-1, *self.obs_dim[self.env_key]['seg'])[indices]
            agent_state_batch = self.agent_state.view(-1, self.obs_dim['state'])[indices]
            action = self.action.view(-1, self.action_dim)[indices]
            expert_action = self.expert_action.view(-1, self.action_dim)[indices]
            value_preds = self.value_preds.view(-1, 1)[indices]
            action_log_probs = self.action_log_probs.view(-1, 1)[indices]
            returns = self.returns.view(-1, 1)[indices]
            obs_dict = {self.env_key: {'rgb': rgb_batch, 'xyz': xyz_batch, 'seg': seg_batch},
                        'state': agent_state_batch}
            data_batch.append([obs_dict, action, adv, value_preds, action_log_probs, returns, expert_action])

        return data_batch
