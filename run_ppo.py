import argparse
from mani_skill_learn.utils.meta import Config
import os
import time
from shutil import copyfile
from mani_skill_learn.env import get_env_info
from mani_skill_learn.networks import build_model
from mani_skill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape
from mani_skill_learn.env.env_utils import build_env
from mani_skill_learn.utils.data import to_torch

from ppo_agent.rollout import RolloutStorage
from ppo_agent.envs import make_vec_envs
from ppo_agent.make_maniskill_env import make_maniskill_env
from ppo_agent.agent import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RL training code')
    # Configurations
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    env_config = Config.fromfile(args.config)
    root_path = env_config.train_mfrl_cfg.root_path
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    local_time = str(time.strftime("%m-%d/%H-%M-%S", time.localtime()))
    root_path = os.path.join(root_path, local_time)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    config_root_path = os.path.join(root_path, "config_files")
    if not os.path.exists(config_root_path):
        os.makedirs(config_root_path)

    src_file = args.config
    dst_file = os.path.join(config_root_path, os.path.basename(src_file))
    copyfile(src_file, dst_file)

    model_root_path = os.path.join(root_path, "models")
    env_config.train_mfrl_cfg.model_save_path = model_root_path
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)

    file_root_path = os.path.join(root_path, "files")
    env_config.train_mfrl_cfg.file_save_path = file_root_path
    if not os.path.exists(file_root_path):
        os.makedirs(file_root_path)

    nn_cfg = env_config.agent.policy_cfg.nn_cfg
    obs_shape, action_shape, action_space = get_env_info(env_config.env_cfg)
    replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)

    nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
    nn_cfg.pop('type')
    env_config.model_cfg['nn_cfg'] = nn_cfg

    from mani_skill_learn.methods.builder import build_mfrl, build_brl, MFRL, BRL
    from mani_skill_learn.apis.train_rl import train_rl
    from mani_skill_learn.env import build_replay
    from mani_skill_learn.utils.torch import load_checkpoint
    import torch.distributed as dist, torch

    # env_make = build_env(env_config.env_cfg)
    env_make = make_maniskill_env(env_config.env_cfg, env_config.replayer_cfg)
    num_process = env_config.rollout_cfg.num_process
    num_state_stack = 0
    num_frame_stack = 0
    device = torch.device('cuda:' + str(env_config.train_mfrl_cfg['device']))
    envs = make_vec_envs(env_make, num_process, file_root_path, device, num_frame_stack, None,
                         num_state_stack)
    obs = envs.reset()
    obs = to_torch(obs, device=device, dtype='float32')

    env_mode = env_config.train_mfrl_cfg['env_mode']
    rollout_cfg = env_config.rollout_cfg
    rollout_cfg['obs_dim'] = obs_shape
    rollout_cfg['action_dim'] = action_shape
    rollout_cfg['env_key'] = env_mode
    rollout = RolloutStorage(**rollout_cfg)
    rollout.to(device)

    rollout.update_obs(obs)

    expert_agent = None

    # for _ in range(10):
    #     print('----act-------')
    cfg = Config.fromfile(env_config.train_mfrl_cfg.expert_cfg_file)
    env_config.model_cfg['policy_head_cfg'] = cfg.agent.policy_cfg.policy_head_cfg

    if env_config.train_mfrl_cfg.expert_cfg_file is not None and env_config.train_mfrl_cfg.expert_cfg_model is not None:
        cfg.agent['obs_shape'] = obs_shape
        cfg.agent['action_shape'] = action_shape
        cfg.agent['action_space'] = action_space
        cfg.resume_from = env_config.train_mfrl_cfg.expert_cfg_model
        expert_agent = build_brl(cfg.agent)
        device_str = 'cuda:' + str(env_config.train_mfrl_cfg.device)
        load_checkpoint(expert_agent, cfg.resume_from, map_location=device_str)
        expert_agent = expert_agent.policy

    #
    train(0, action_shape, env_config.model_cfg, env_config.train_mfrl_cfg, False, expert_agent, envs, rollout,
          rollout_cfg.num_steps, num_process)
