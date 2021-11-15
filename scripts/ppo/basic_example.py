import gym

import sys
sys.path.append('ManiSkill-Learn')
print('sys.path:', sys.path)

import mani_skill.env
import numpy as np
import cv2
import torch
import mani_skill
# print('mani_skill.__path__: ', mani_skill.__path__)
import ppo_agent
print('ppo_agent.__path__: ', ppo_agent.__path__)
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

env = make_gym_env(env_name, obs_mode='pointcloud')
print('after make env')
# env2 = gym.make('OpenCabinetDoor_1000-v0')
# print('after make env2')

# full environment list can be found in available_environments.txt

# env.set_env_mode(obs_mode='state', reward_type='sparse')
# env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
# env2.set_env_mode(obs_mode='rgbd', reward_type='sparse')
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
# print(env.observation_space)  # this shows the structure of the observation, openai gym's format
# print(env.action_space)  # this shows the action space, openai gym's format


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

from ppo_agent.model import Model, TestModel

obs = env.reset()
obs = to_torch(obs, device=device, dtype='float32')
for key in obs:
    if not isinstance(obs[key], dict):
        obs[key] = obs[key].unsqueeze(0)
    else:
        for sub_key in obs[key]:
            obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)
# local_model = Model(**model_cfg)
local_model = TestModel(**model_cfg)
pre_action = local_model.test_act(obs)
print('pre_action', pre_action)
# model_path = '/home/quan/maniskill_model/11-03/20-27-13/models/ppo_model_1000.pt'
# failure
# model_path = '/home/quan/maniskill_model/11-10/23-22-20/models/ppo_model_500.pt'
# failure
# model_path = '/home/quan/maniskill_model/11-10/23-27-51/models/ppo_model_500.pt'
# model_path = '/home/quan/maniskill_model/11-11/21-50-40/models/ppo_model_500.pt'
model_path = '/home/quan/maniskill_model/11-11/21-50-40/models/ppo_model_1000.pt'
learnt_model = torch.load(model_path, map_location=device)
local_model.load_state_dict(learnt_model.state_dict())
local_model.trainable = False
local_model.eval()
# local_model.to_device(device)
local_model.to(device)
del learnt_model

for level_idx in range(0, 5):  # level_idx is a random seed
    # obs = env.reset(level=level_idx)
    obs = env.reset()
    obs = to_torch(obs, device=device, dtype='float32')
    for key in obs:
        if not isinstance(obs[key], dict):
            obs[key] = obs[key].unsqueeze(0)
        else:
            for sub_key in obs[key]:
                obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)

    # env.seed(np.random.randint(0, 10000))

    # obs2 = env2.reset(level=level_idx)

    # print(obs['obs']['point_cloud'].keys())
    # print(obs)
    # print('#### Level {:d}'.format(level_idx))
    for i_step in range(100000):
        # print(i_step)
        env.render('human')  # a display is required to use this function, rendering will slower the running speed
        # env2.render('env2')
        # obs['agent'] = obs['state']
        # qpos = osc_interface.get_robot_qpos_from_obs(obs)

        # action = np.zeros_like(env.action_space.sample())

        # joint_action = action
        # os_action, null_action = osc_interface.joint_space_to_operational_space_and_null_space(qpos, joint_action)
        # joint_action_rec = osc_interface.operational_space_and_null_space_to_joint_space(qpos, os_action,
        #                                                                                  null_action)
        # epsilon = 1E-6
        # if np.max(np.abs(joint_action_rec - action)) > epsilon:
        #     print('Reconstruct Error!', joint_action_rec, action)
        #     exit(-1)
        #
        # # Example 2: Move end effector along a specific direction
        # hand_forward = np.zeros(osc_interface.osc_dim)
        # extra_dim = len(osc_interface.osc_extra_joints)
        #
        # dim = 3  # move along x direction in end effector's frame
        # # hand_forward[extra_dim + dim:extra_dim + dim + 1] = 0.1  # 0.1 is the target velocity in velocity controller
        # hand_forward[extra_dim + dim:extra_dim + dim + 1] = 0 # 0.1 is the target velocity in velocity controller
        # # hand_forward[extra_dim + dim + 6:extra_dim + dim + 7] = 0.1  # this is left arm when the task needs two arms
        # forward_action = osc_interface.operational_space_and_null_space_to_joint_space(
        #     qpos, hand_forward, action[:osc_interface.null_space_dim])
        # action = forward_action
        # print('action:', action)
        # action[0] = 1.0
        # action[1] = 1.0
        # action[2] = 1.0
        # action[3] = 1.0

        # action[4] = 1.0
        # action[5] = 1.0
        # action[6] = 1.0
        # action[7] = 1.0
        # action[8] = 1.0
        # action[9] = 1.0

        # action[10] = 1.0
        # action[11] = 1.0
        # action[12] = 1.0
        # _, action, _, _ = local_model.act(obs)
        _, action, _, _ = local_model.test_act(obs)
        print('mode action is ', action)
        action = action.squeeze().cpu().numpy()
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)  # take a random action
        if done:
            if info['eval_info']['success']:
                print('success!!!')
            else:
                print('failed...')

        # print('info:', info)
        obs = to_torch(obs, device=device, dtype='float32')
        for key in obs:
            if not isinstance(obs[key], dict):
                obs[key] = obs[key].unsqueeze(0)
            else:
                for sub_key in obs[key]:
                    obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)

        # rollout.insert(sampled_batch, action)
        # obs = dict(pointcloud=sampled_batch['pointcloud'], state=sampled_batch['state'])
        # sampled_batch = dict(obs=obs, actions=action)
        # sampled_batch = to_torch(sampled_batch, device=device, dtype='float32')

        # print(sampled_batch['obs']['pointcloud']['rgb'].size())
        # for key in sampled_batch:
        #     if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
        #         sampled_batch[key] = sampled_batch[key][..., None]
        # print(sampled_batch.keys())
        # obs_features = backbone(sampled_batch['obs'])
        #
        # if i_step % (128 + 1) == 0:
        #     obs_state = rollout.feed_forward_generator(None)
        #     for batch_data in obs_state:
        #         obs_features = backbone(batch_data['obs'])
        #         print('obs_features.size:', obs_features.size())

        # # env.render('env1')
        # # env2.render('env2')
        # rgb_image = np.array(obs['rgbd']['rgb'])
        # rgb_image_left = rgb_image[:, :, 0:3]
        # rgb_image_center = rgb_image[:, :, 3:6]
        # rgb_image_right = rgb_image[:, :, 6:9]
        # rgb_image_mask = np.array(obs['rgbd']['seg'])
        # rgb_image_left_mask = rgb_image_mask[:, :, 0:3]
        # rgb_image_center_mask = rgb_image_mask[:, :, 3:6]
        # rgb_image_right_mask = rgb_image_mask[:, :, 6:9]
        #
        # rgb_image = np.hstack([rgb_image_left * rgb_image_left_mask, rgb_image_center * rgb_image_center_mask,
        #                        rgb_image_right * rgb_image_right_mask])
        # cv2.imshow('rgb_center', rgb_image)
        # cv2.waitKey(1)
        #
        #
        # action2 = env2.action_space.sample()
        # obs, reward, done, info = env2.step(action2)  # take a random action
        # # env.render('env1')
        # # env2.render('env2')
        # rgb_image = np.array(obs['rgbd']['rgb'])
        # rgb_image_left = rgb_image[:, :, 0:3]
        # rgb_image_center = rgb_image[:, :, 3:6]
        # rgb_image_right = rgb_image[:, :, 6:9]
        # rgb_image_mask = np.array(obs['rgbd']['seg'])
        # rgb_image_left_mask = rgb_image_mask[:, :, 0:3]
        # rgb_image_center_mask = rgb_image_mask[:, :, 3:6]
        # rgb_image_right_mask = rgb_image_mask[:, :, 6:9]
        #
        # rgb_image = np.hstack([rgb_image_left * rgb_image_left_mask, rgb_image_center * rgb_image_center_mask,
        #                        rgb_image_right * rgb_image_right_mask])
        # # print(rgb_image.shape)
        # # print(rgb_image)
        # # rgb_image = rgb_image.transpose()
        # # image = np.zeros(400, 160, 3)
        #
        # cv2.imshow('rgb_center2', rgb_image)
        # cv2.waitKey(1)
        # print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
        if done:
            break
env.close()
# env2.close()
