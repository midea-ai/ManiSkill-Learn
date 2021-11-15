import gym
import sys
import mani_skill.env
import numpy as np
import cv2
import torch
import mani_skill
import ppo_agent
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape
from mani_skill_learn.env import get_env_info
from mani_skill_learn.networks.builder import POLICYNETWORKS, build_backbone, build_dense_head
from mani_skill_learn.utils.meta import Config
from ppo_agent.rollout import RolloutStorage
from mani_skill_learn.env import make_gym_env
from ppo_agent.model import PointNetWithInstanceInfoV0
from mani_skill.utils.osc import OperationalSpaceControlInterface
from gym import envs

# env = gym.make('OpenCabinetDrawer-v0')
env_mode = 'pointcloud'
env_name = 'OpenCabinetDrawer_1045-v0'


# env_config_file = 'configs/bc/mani_skill_point_cloud_transformer.py'
env_config_file = 'configs/ppo/mani_skill_point_cloud_transformer.py'
env_config = Config.fromfile(env_config_file)
print('config:', env_config.env_cfg)
obs_shape, action_shape, action_space = get_env_info(env_config.env_cfg)

env = gym.make(env_name)
env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')

# nn_cfg = env_config['agent']['policy_cfg']['nn_cfg']
nn_cfg = env_config.agent.policy_cfg.nn_cfg

# obs_shape, action_shape, action_space = get_env_info(env_config.env_cfg)

replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)

nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
# backbone = build_backbone(nn_cfg)
nn_cfg.pop('type')

backbone = PointNetWithInstanceInfoV0(**nn_cfg)
device = torch.device('cuda:0')
# print(backbone)
backbone = backbone.to(device)

# rollout = RolloutStorage(128, 4, obs_shape, action_space)
# rollout.to(device)
# print('nn_cfg:', nn_cfg)
osc_interface = OperationalSpaceControlInterface(env_name)

env_config.model_cfg['nn_cfg'] = nn_cfg
model_cfg = env_config.model_cfg
# model_cfg['obs_dim'] = obs_shape
model_cfg['action_dim'] = action_shape
# model_cfg['action_space'] = 'continuous'
model_cfg['trainable'] = False

cfg = Config.fromfile(env_config.train_mfrl_cfg.expert_cfg_file)
env_config.model_cfg['policy_head_cfg'] = cfg.agent.policy_cfg.policy_head_cfg

from ppo_agent.model import Model

obs = env.reset()

# obs = to_torch(obs, device=device, dtype='float32')
# for key in obs:
#     if not isinstance(obs[key], dict):
#         obs[key] = obs[key].unsqueeze(0)
#     else:
#         for sub_key in obs[key]:
#             obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)
# local_model = Model(**model_cfg)
# # model_path = '/home/quan/maniskill_model/11-03/20-27-13/models/ppo_model_1000.pt'
# # failure
# # model_path = '/home/quan/maniskill_model/11-10/23-22-20/models/ppo_model_500.pt'
# # failure
# model_path = '/home/quan/maniskill_model/11-10/23-27-51/models/ppo_model_500.pt'
# # model_path = '/home/quan/maniskill_model/11-11/21-50-40/models/ppo_model_500.pt'
# # model_path = '/home/quan/maniskill_model/11-11/21-50-40/models/ppo_model_1500.pt'
# learnt_model = torch.load(model_path, map_location=device)
# local_model.load_state_dict(learnt_model.state_dict())
# local_model.trainable = False
# local_model.eval()
# # local_model.to_device(device)
# local_model.to(device)
# del learnt_model
from user_solution_ppo import UserPolicy
local_model = UserPolicy(env_name)
for level_idx in range(0, 5):  # level_idx is a random seed
    # obs = env.reset(level=level_idx)
    obs = env.reset()
    # obs = to_torch(obs, device=device, dtype='float32')
    # for key in obs:
    #     if not isinstance(obs[key], dict):
    #         obs[key] = obs[key].unsqueeze(0)
    #     else:
    #         for sub_key in obs[key]:
    #             obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)

    for i_step in range(100000):
        # print(i_step)
        # env.render('human')  # a display is required to use this function, rendering will slower the running speed

        action = local_model.act(obs)
        print('mode action is ', action)
        # action = action.squeeze().cpu().numpy()
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)  # take a random action
        if done:
            if info['eval_info']['success']:
                print('success!!!')
            else:
                print('failed...')

        # obs = to_torch(obs, device=device, dtype='float32')
        # for key in obs:
        #     if not isinstance(obs[key], dict):
        #         obs[key] = obs[key].unsqueeze(0)
        #     else:
        #         for sub_key in obs[key]:
        #             obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)

        if done:
            break
env.close()
# env2.close()
