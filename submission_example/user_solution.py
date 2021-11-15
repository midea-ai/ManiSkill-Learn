import pathlib
from collections import deque

import gym
import numpy as np

from mani_skill_learn.env import get_env_info
from mani_skill_learn.env.observation_process import process_mani_skill_base
from mani_skill_learn.methods.builder import build_brl
from mani_skill_learn.utils.data import to_np, unsqueeze
from mani_skill_learn.utils.meta import Config
from mani_skill_learn.utils.torch import load_checkpoint
from ppo_agent.model import Model

import os

class ObsProcess:
    # modified from SapienRLWrapper
    def __init__(self, env, obs_mode, stack_frame=1):
        """
        Stack k last frames for point clouds or rgbd
        """
        self.env = env
        self.obs_mode = obs_mode
        self.stack_frame = stack_frame
        self.buffered_data = {}

    def _update_buffer(self, obs):
        for key in obs:
            if key not in self.buffered_data:
                self.buffered_data[key] = deque([obs[key]] * self.stack_frame, maxlen=self.stack_frame)
            else:
                self.buffered_data[key].append(obs[key])

    def _get_buffer_content(self):
        axis = 0 if self.obs_mode == 'pointcloud' else -1
        return {key: np.concatenate(self.buffered_data[key], axis=axis) for key in self.buffered_data}

    def process_observation(self, observation):
        if self.obs_mode == "state":
            return observation
        observation = process_mani_skill_base(observation, self.env)
        visual_data = observation[self.obs_mode]
        self._update_buffer(visual_data)
        visual_data = self._get_buffer_content()
        state = observation['agent']
        # Convert dict of array to list of array with sorted key
        ret = {}
        ret[self.obs_mode] = visual_data
        ret['state'] = state
        return ret


class BasePolicy(object):
    def __init__(self, opts=None):
        self.obs_mode = 'pointcloud'

    def act(self, observation):
        raise NotImplementedError()

    def reset(self):  # if you use an RNN-based policy, you need to implement this function
        pass


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.env = gym.make(env_name)
        self.obs_mode = 'pointcloud'  # remember to set this!
        self.env.set_env_mode(obs_mode=self.obs_mode)
        self.stack_frame = 1

        # cfg_path = str(pathlib.Path('./configs/bc/mani_skill_point_cloud_transformer.py').resolve())
        cfg_path = os.path.join(os.path.dirname(__file__), '../configs/cql/mani_skill_point_cloud_transformer.py')
        cfg_path = str(pathlib.Path(cfg_path).resolve())
        cfg = Config.fromfile(cfg_path)
        cfg.env_cfg['env_name'] = env_name
        obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        cfg.agent['obs_shape'] = obs_shape
        cfg.agent['action_shape'] = action_shape
        cfg.agent['action_space'] = action_space

        self.agent = build_brl(cfg.agent)

        env_config_file = 'configs/ppo/mani_skill_point_cloud_transformer.py'
        env_config = Config.fromfile(env_config_file)
        nn_cfg = env_config.agent.policy_cfg.nn_cfg

        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)

        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        nn_cfg.pop('type')
        env_config.model_cfg['nn_cfg'] = nn_cfg
        model_cfg = env_config.model_cfg
        model_cfg['action_dim'] = action_shape
        model_cfg['trainable'] = False
        cfg = Config.fromfile(env_config.train_mfrl_cfg.expert_cfg_file)
        env_config.model_cfg['policy_head_cfg'] = cfg.agent.policy_cfg.policy_head_cfg



        local_model = Model(**model_cfg)
        # model_path = '/home/quan/maniskill_model/11-03/20-27-13/models/ppo_model_1000.pt'
        # failure
        # model_path = '/home/quan/maniskill_model/11-10/23-22-20/models/ppo_model_500.pt'
        # failure
        model_path = '/home/quan/maniskill_model/11-10/23-27-51/models/ppo_model_500.pt'
        # model_path = '/home/quan/maniskill_model/11-11/21-50-40/models/ppo_model_500.pt'
        # model_path = '/home/quan/maniskill_model/11-11/21-50-40/models/ppo_model_1500.pt'
        learnt_model = torch.load(model_path, map_location=device)
        local_model.load_state_dict(learnt_model.state_dict())
        local_model.trainable = False
        local_model.eval()
        # local_model.to_device(device)
        local_model.to(device)
        del learnt_model

        if env_name.find('Bucket') >= 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), '../work_dirs/cql_transformer_bucket/CQL/models/model_115000.ckpt')
        if env_name.find('Chair') >= 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), '../work_dirs/cql_transformer_chair/CQL/models/model_115000.ckpt')
        if env_name.find('Door') >= 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), '../work_dirs/cql_transformer_door/CQL/models/model_90000.ckpt')
        if env_name.find('Drawer') >= 0:
            # ckpt_path = os.path.join(os.path.dirname(__file__), '../work_dirs/cql_transformer_drawer/CQL/models/model_90000.ckpt')
            ckpt_path = os.path.join(os.path.dirname(__file__), '../work_dirs/PPO_drawer/11-10/23-27-51/models/ppo_model_500.pt')
        # load_checkpoint(self.agent,
        #     str(pathlib.Path('./example_mani_skill_data/OpenCabinetDrawer_1045_link_0-v0_PN_Transformer.ckpt').resolve()),
        #     map_location='cpu'
        # )
        load_checkpoint(self.agent,
            str(pathlib.Path(ckpt_path).resolve()),
            map_location='cpu'
        )
        self.agent.to('cuda')  # dataparallel not done here
        self.agent.eval()

        self.obsprocess = ObsProcess(self.env, self.obs_mode, self.stack_frame)

    def act(self, observation):
        ##### Replace with your code
        observation = self.obsprocess.process_observation(observation)
        return to_np(self.agent(unsqueeze(observation, axis=0), mode='eval'))[0]
