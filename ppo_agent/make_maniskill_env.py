from mani_skill_learn.env.env_utils import build_env
from gym.core import Wrapper
from ppo_agent.replay import ReplayAll
import numpy as np


class ManiSkillWrapper(Wrapper):
    def __init__(self, env, replay_args):
        super(ManiSkillWrapper, self).__init__(env)
        self.threshold_reward = replay_args.pop('threshold_reward')
        self.replayer = ReplayAll(**replay_args)
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.value_rollouts = []
        # self.success = False
        self.demo = None
        self.demo_value = None
        self.env_reward = 0.0
        self.env_reward_no_D = 0.0

    def step(self, action_):
        action, info_in = action_
        out = self.replayer.replay_step(action, info_in['value'])
        info = {}
        if len(out) == 1:  # no reply, interact with env
            obs, reward, done, info = self.env.step(action)
            self.obs_rollouts.append(obs)
            self.rews_rollouts.append(reward)
            self.actions_rollouts.append(action)
            # print('value:', info_in['value'])
            self.value_rollouts.append(info_in['value'])
            info['true_action'] = False
            if done and info['eval_info']['success'] is True:
                self.success = True
            self.env_reward += reward
            self.env_reward_no_D += reward

        else:  # get from replay
            action, obs, reward, done = out
            info['true_action'] = True
            info['eval_info'] = {}
            info['eval_info']['open_enough'] = done
            info['eval_info']['success'] = done
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env_reward = 0.0
        self.env_reward_no_D = 0.0
        self.steps = 0
        self.demo = None
        if (len(self.actions_rollouts) > 0):
            if np.mean(self.rews_rollouts) > self.threshold_reward or self.success:
                # print('add one trajectory to D_r')
                self.demo = {
                    'obs': np.array(self.obs_rollouts),
                    'rewards': np.array(self.rews_rollouts),
                    'actions': np.array(self.actions_rollouts),
                    'values': np.array(self.value_rollouts)
                }
            else:
                if self.replayer.sample_mode == "value" and (max(self.value_rollouts) > self.replayer.min_value) or \
                        self.replayer.sample_mode == "reward" and (
                        np.mean(self.value_rollouts) > self.replayer.min_value):
                    # print('add one trajectory to D_v')
                    self.demo_value = {
                        'obs': np.array(self.obs_rollouts),
                        'rewards': np.array(self.rews_rollouts),
                        'actions': np.array(self.actions_rollouts),
                        'values': np.array(self.value_rollouts)
                    }
        if self.demo and not self.replayer.replay:
            self.replayer.add_demo(self.demo)
        if self.demo_value and not self.replayer.replay:
            self.replayer.add_demo_value(self.demo_value)
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.value_rollouts = []
        self.replayer.reset()
        self.success = False
        return self.env.reset(**kwargs)


def make_maniskill_env(env_args, replay_args):
    def make_env(rank):
        def _thunk():
            seed = rank
            env = build_env(env_args)
            env = ManiSkillWrapper(env, replay_args)

            return env

        return _thunk

    return make_env
