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

import os

class ObsProcess:
    # modified from Saself.agentenRLWrapper
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
        # cfg_path = os.path.join(os.path.dirname(__file__), './configs/cql/mani_skill_point_cloud_transformer.py')
        # cfg_path = os.path.join(os.path.dirname(__file__), './configs/bc/mani_skill_point_cloud_transformer.py')
        # cfg_path = os.path.join(os.path.dirname(__file__), './configs/bc/mani_skill_point_cloud_transformer2.py')
        # cfg_path = str(pathlib.Path(cfg_path).resolve())
        # cfg = Config.fromfile(cfg_path)
        # cfg.env_cfg['env_name'] = env_name
        # obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        # cfg.agent['obs_shape'] = obs_shape
        # cfg.agent['action_shape'] = action_shape
        # cfg.agent['action_space'] = action_space

        self.agents = []
        if env_name.find('Door') >= 0:
            matrix_index = [i for i in range(20)]
            model_paths = [
                './work_dirs/bc_pointnet_transformer_door10_0/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_1/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_2/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_3/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_4/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_5/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_6/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_7/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_8/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_9/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_10/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_11/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_12/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_13/BC/models/model_130000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_14/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_15/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_16/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_17/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_18/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_door10_19/BC/models/model_120000.ckpt',
            ]
            config7 = False #True
            model7_path = './work_dirs/bc_pointnet_transformer_door7/BC/models/model_150000.ckpt'
        elif env_name.find('Drawer') >= 0:
            matrix_index = [i for i in range(20)]
            model_paths = [
                './work_dirs/bc_pointnet_transformer_drawer10_0/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_1/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_2/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_3/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_4/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_5/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_6/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_7/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_8/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_9/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_10/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_11/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_12/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_13/BC/models/model_130000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_14/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_15/BC/models/model_130000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_16/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_17/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_18/BC/models/model_130000.ckpt',
                './work_dirs/bc_pointnet_transformer_drawer10_19/BC/models/model_120000.ckpt',
            ]
            config7 = False #True
            model7_path = './work_dirs/bc_pointnet_transformer_drawer7/BC/models/model_120000.ckpt'
        elif env_name.find('Bucket') >= 0:
            matrix_index = [i for i in range(20)]
            model_paths = [
                './work_dirs/bc_pointnet_transformer_bucket10_0/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_1/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_2/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_3/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_4/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_5/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_6/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_7/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_8/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_9/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_10/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_11/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_12/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_13/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_14/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_15/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_16/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_17/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_18/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_bucket10_19/BC/models/model_140000.ckpt',
            ]
            config7 = False #True #False
            model7_path = './work_dirs/bc_pointnet_transformer_bucket7/BC/models/model_30000.ckpt'
        elif env_name.find('Chair') >= 0:
            matrix_index = [i for i in range(20)]
            model_paths = [
                './work_dirs/bc_pointnet_transformer_chair10_0/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_1/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_2/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_3/BC/models/model_130000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_4/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_5/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_6/BC/models/model_100000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_7/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_8/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_9/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_10/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_11/BC/models/model_110000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_12/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_13/BC/models/model_150000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_14/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_15/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_16/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_17/BC/models/model_140000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_18/BC/models/model_120000.ckpt',
                './work_dirs/bc_pointnet_transformer_chair10_19/BC/models/model_100000.ckpt',
            ]
            config7 = False #True #False
            model7_path = './work_dirs/bc_pointnet_transformer_chair7/BC/models/model_150000.ckpt'

        for idx, i in enumerate(matrix_index):
            cfg_path = os.path.join(os.path.dirname(__file__), './configs/bc/mani_skill_point_cloud_transformer10.py')
            cfg_path = str(pathlib.Path(cfg_path).resolve())
            cfg = Config.fromfile(cfg_path)
            cfg.env_cfg['env_name'] = env_name
            obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
            cfg.agent['obs_shape'] = obs_shape
            cfg.agent['action_shape'] = action_shape
            cfg.agent['action_space'] = action_space
            cfg.agent['policy_cfg']['nn_cfg']['matrix_index'] = i

            cur_agent = build_brl(cfg.agent)
            ckpt_path = os.path.join(os.path.dirname(__file__), model_paths[idx])

            load_checkpoint(cur_agent,
                str(pathlib.Path(ckpt_path).resolve()),
                map_location='cpu'
            )
            cur_agent.to('cuda')  # dataparallel not done here
            cur_agent.eval()
            self.agents.append(cur_agent)
        
        if config7:
            cfg_path = os.path.join(os.path.dirname(__file__), './configs/bc/mani_skill_point_cloud_transformer7.py')
            cfg_path = str(pathlib.Path(cfg_path).resolve())
            cfg = Config.fromfile(cfg_path)
            cfg.env_cfg['env_name'] = env_name
            obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
            cfg.agent['obs_shape'] = obs_shape
            cfg.agent['action_shape'] = action_shape
            cfg.agent['action_space'] = action_space
            cfg.agent['policy_cfg']['nn_cfg']['matrix_index'] = -1

            cur_agent = build_brl(cfg.agent)
            ckpt_path = os.path.join(os.path.dirname(__file__), model7_path)

            load_checkpoint(cur_agent,
                str(pathlib.Path(ckpt_path).resolve()),
                map_location='cpu'
            )
            cur_agent.to('cuda')  # dataparallel not done here
            cur_agent.eval()
            self.agents.append(cur_agent)
                
        # self.agent = build_brl(cfg.agent)
        # if env_name.find('Bucket') >= 0:
        #     ckpt_path = os.path.join(os.path.dirname(__file__), './work_dirs/cql_transformer_bucket/CQL/models/model_115000.ckpt')
        # if env_name.find('Chair') >= 0:
        #     ckpt_path = os.path.join(os.path.dirname(__file__), './work_dirs/cql_transformer_chair/CQL/models/model_115000.ckpt')
        # if env_name.find('Door') >= 0:
            # ckpt_path = os.path.join(os.path.dirname(__file__), './work_dirs/base_bc_point_transformer_door/BC/models/model_140000.ckpt')
            # ckpt_path = os.path.join(os.path.dirname(__file__), './work_dirs/bc_pointnet_transformer_door3/BC/models/model_5000.ckpt')
            # ckpt_path = os.path.join(os.path.dirname(__file__), './work_dirs/bc_pointnet_transformer_door3/BC/models/model_25000.ckpt')
        # if env_name.find('Drawer') >= 0:
        #     ckpt_path = os.path.join(os.path.dirname(__file__), './work_dirs/cql_transformer_drawer/CQL/models/model_90000.ckpt')
        # load_checkpoint(self.agent,
        #     str(pathlib.Path('./example_mani_skill_data/OpenCabinetDrawer_1045_link_0-v0_PN_Transformer.ckpt').resolve()),
        #     map_location='cpu'
        # )
        # load_checkpoint(self.agent,
        #     str(pathlib.Path(ckpt_path).resolve()),
        #     map_location='cpu'
        # )
        # self.agent.to('cuda')  # dataparallel not done here
        # self.agent.eval()

        self.lstm_obs = []

        self.obsprocess = ObsProcess(self.env, self.obs_mode, self.stack_frame)

    # def act(self, observation):
    #     ##### Replace with your code
    #     observation = self.obsprocess.process_observation(observation)
    #     return to_np(self.agent(unsqueeze(observation, axis=0), mode='eval'))[0]

    def reset(self):
        self.lstm_obs = []
        return super().reset()

    def act(self, observation):
        ##### Replace with your code
        # observation = self.obsprocess.process_observation(observation)
        # return to_np(self.agent(unsqueeze(observation, axis=0), mode='eval'))[0]

        obs = self.obsprocess.process_observation(observation)
        action_list = []
        
        if len(self.lstm_obs) < self.agents[0].lstm_len:
                self.lstm_obs.append(obs)
        else:
            for i in range(self.agents[0].lstm_len - 1):
                self.lstm_obs[i] = self.lstm_obs[i + 1]
            self.lstm_obs[-1] = obs
    
        for cur_agent in self.agents:
            if cur_agent.lstm_len == 1 or len(self.lstm_obs) < cur_agent.lstm_len:
                action = to_np(cur_agent(unsqueeze(obs, axis=0), mode='eval'))[0]
                action_list.append(action)
            else:
                # for k in merge_obs:
                #     print("k: %s; v: %s" % (k, merge_obs[k]))
                state_list = []
                xyz_list = []
                rgb_list = []
                seg_list = []
                for i in range(cur_agent.lstm_len):
                    state_list.append(self.lstm_obs[i]['state'])
                    xyz_list.append(self.lstm_obs[i]['pointcloud']['xyz'])
                    rgb_list.append(self.lstm_obs[i]['pointcloud']['rgb'])
                    seg_list.append(self.lstm_obs[i]['pointcloud']['seg'])

                    # k = 'state'
                    # merge_obs[k] = [merge_obs[k], lstm_obs[i+1].get(k)]
                    # k = 'pointcloud'
                    # merge_obs[k] = {sub_k: [merge_obs[k][sub_k], lstm_obs[i+1][k][sub_k]] for sub_k in merge_obs[k]}
                    # merge_obs = {k: [merge_obs[k], lstm_obs[i+1].get(k)] for k in merge_obs}
                
                merge_obs = self.lstm_obs[0]
                merge_obs['state'] = np.stack(state_list)
                # print("state: %s" % (str(merge_obs['state'].shape)))
                # k = 'state'
                # # merge_obs = {k: np.stack(merge_obs[k]) for k in merge_obs}
                # merge_obs[k] = np.stack(merge_obs[k])
                # print("k: %s; v: %s" % (k, merge_obs[k].shape))

                k = 'pointcloud'
                # merge_obs[k] = {sub_k: np.stack(merge_obs[k][sub_k]) for sub_k in merge_obs[k]}
                merge_obs[k]['xyz'] = np.stack(xyz_list)
                merge_obs[k]['rgb'] = np.stack(rgb_list)
                merge_obs[k]['seg'] = np.stack(seg_list)
                # for sub_k in merge_obs[k]:
                    # print("sub_k: %s; v: %s" % (str(sub_k), str(merge_obs[k][sub_k].shape)))
                # action = to_np(self.agent(unsqueeze(merge_obs, axis=0), mode=self.sample_mode))[0]

                action = to_np(cur_agent(merge_obs, mode='eval'))[0]
                # print("cur action: %s" % (str(action.shape)))
                action_list.append(action)
        
        action_list = np.stack(action_list)
        action = np.mean(action_list, axis=0)
        
        return action
