import random
import numpy as np
import torch
import torch.optim as optim
from ppo_agent.rollout import RolloutStorage
from ppo_agent.model import Model
import os
# from mani_skill_learn.env import make_gym_env
from mani_skill_learn.env.env_utils import build_env
from mani_skill_learn.utils.data import to_torch
import time


def train(rank, action_dim, model_cfg, train_cfg, distributed, expert_agent, env, rollout, num_steps, num_processes):
    seed = rank + 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_episode = train_cfg['total_episode']
    use_adv_norm = train_cfg['use_adv_norm']
    ppo_epoch = train_cfg['ppo_epoch']
    clip = train_cfg['clip']
    value_coeff = train_cfg['value_coeff']
    action_coeff = train_cfg['action_coeff']
    ent_coeff = train_cfg['ent_coeff']
    lr = train_cfg['lr']
    model_save_rate = train_cfg['model_save_rate']
    model_save_path = train_cfg['model_save_path']
    file_save_path = train_cfg['file_save_path']
    warm_up = train_cfg['warm_up']

    device = torch.device('cuda:' + str(train_cfg['device']))
    # env_mode = train_cfg['env_mode']
    #
    # # env_name = env_cfg['env_name']
    # env = build_env(env_cfg)
    # # env = make_gym_env(env_name, obs_mode=env_mode)
    # # env.set_env_mode(obs_mode=env_mode, reward_type='dense')
    # obs = env.reset(level=0)
    # rollout_cfg['obs_dim'] = obs_dim
    # rollout_cfg['action_dim'] = action_dim
    # rollout_cfg['env_key'] = env_mode
    # rollout = RolloutStorage(**rollout_cfg)
    # rollout.to(device)
    # rollout.update_obs(obs)
    #
    # obs = to_torch(obs, device=device, dtype='float32')
    # for key in obs:
    #     if not isinstance(obs[key], dict):
    #         obs[key] = obs[key].unsqueeze(0)
    #     else:
    #         for sub_key in obs[key]:
    #             obs[key][sub_key] = obs[key][sub_key].unsqueeze(0)

    # model_cfg['obs_dim'] = obs_dim
    model_cfg['action_dim'] = action_dim
    # model_cfg['action_space'] = 'continuous'
    model_cfg['trainable'] = True
    # local_model = ContinuousPolicy(**model_cfg)
    # local_model.to(device)

    local_model = Model(**model_cfg)

    if expert_agent and warm_up:
        local_model.load_state_dict(expert_agent.state_dict(), strict=False)
        local_model.load_state_dict(expert_agent.backbone.state_dict(), strict=False)
        del expert_agent
        # for name, p in expert_agent.backbone.named_parameters():
        #     print('in expert---> name: ', name, p.data.mean(), 'p.requires_grad', p.requires_grad)

    if train_cfg['pretrain'] is True and train_cfg['pretrain_cfg_model'] is not None:
        pretrain_model = torch.load(train_cfg['pretrain_cfg_model'], map_location=device)
        local_model.load_state_dict(pretrain_model.state_dict())
        del pretrain_model
        print('successfully load pretrained model from ', train_cfg['pretrain_cfg_model'])
    local_model.to(device)

    # local_model.print_parameters()

    # num_steps = rollout_cfg['num_steps']
    overall_loss = []
    overall_reward = []
    episode = 0
    steps = 0
    sum_reward = 0
    av_reward = []

    if not distributed:
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
    st_time = time.time()
    # infos_in = [{'demo_value_in': [], 'demo_in': [], 'value': None} for i in range(num_processes)]
    # todo: debug ?
    infos_in = [{'demo_value_in': [], 'demo_in': [], 'value': 0} for i in range(num_processes)]

    while episode < total_episode:
        expert_action = None
        # env.render('human')
        with torch.no_grad():
            obs = rollout.get_obs()

            value, action, log_probs, expert_action1 = local_model.act(obs)

        for ii in range(len(infos_in)):
            infos_in[ii]['value'] = value[ii].cpu().numpy()
        action_ = (action, infos_in)
        obs, reward, done, infos = env.step(action_)  # take a random action
        # expert_action1 = (expert_action1, infos_in)
        # obs, reward, done, infos = env.step(expert_action1)  # take an expert action
        # print('success:', info[0]['eval_info']['success'], '; open_enough', info[0]['eval_info']['open_enough'])
        # print(done)


        with torch.no_grad():
            for ii, info in enumerate(infos):
                if info['true_action'] is True:
                    # replay
                    log_probs[ii] = 0

        obs = to_torch(obs, device=device, dtype='float32')
        sum_reward += reward
        av_reward.append(reward)
        masks = torch.tensor(1 - done).unsqueeze(-1)
        rollout.insert(obs, action, value, log_probs, masks, reward, expert_action)
        # if int(done[0]) == 1:
        #     print('avg_reward:', np.mean(av_reward))
        #     av_reward = []
        steps += 1

        if steps % num_steps == 0:

            with torch.no_grad():
                next_value = local_model.get_value(obs)
                masks = masks.to(device)
                next_value = next_value * masks

            rollout.compute_returns(next_value)
            advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
            if use_adv_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            sum_action_loss = sum_value_loss = sum_entropy_loss = sum_total_loss = 0
            for _ in range(ppo_epoch):
                data_generator = rollout.feed_forward_generator(advantages)
                for samples in data_generator:
                    state_batch, action_batch, adv_batch, value_preds_batch, log_probs_batch, return_batch, expert_action_batch = samples

                    cur_values, cur_action_log_probs, entropy = local_model.evaluate_actions(state_batch, action_batch)
                    ratio = torch.exp(cur_action_log_probs - log_probs_batch)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_batch
                    action_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = value_preds_batch + (cur_values - value_preds_batch).clamp(-clip, clip)
                    value_losses = (cur_values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()

                    entropy_loss = entropy.mean()
                    # expert_action_l
                    total_loss = action_coeff * action_loss + value_coeff * value_loss - ent_coeff * entropy_loss
                    sum_action_loss += action_loss.item()
                    sum_value_loss += value_loss.item()
                    sum_entropy_loss += entropy_loss.item()
                    sum_total_loss += total_loss.item()
                    local_model.zero_grad()
                    if not distributed:
                        optimizer.zero_grad()

                    total_loss.backward()

                    # local_model.print_grad()
                    if not distributed:
                        optimizer.step()

            sum_total_loss /= ppo_epoch
            sum_action_loss /= ppo_epoch
            sum_value_loss /= ppo_epoch
            sum_entropy_loss /= ppo_epoch
            overall_loss.append([sum_total_loss, action_coeff * sum_action_loss, value_coeff * sum_value_loss,
                                 ent_coeff * sum_entropy_loss])
            overall_reward.append(sum_reward[0].item() / num_steps)
            episode += 1
            ed_time = time.time()
            hour = (ed_time - st_time) // 3600
            minute = ((ed_time - st_time) % 3600) // 60
            second = (ed_time - st_time) % 60
            print(hour, ':', minute, ':', second, ' in episode ', episode, '[tt_loss:', round(sum_total_loss, 3),
                  ', a_loss:',
                  round(action_coeff * sum_action_loss, 3), ', v_loss', round(value_coeff * sum_value_loss, 3),
                  ', ent_loss',
                  round(ent_coeff * sum_entropy_loss, 3), '], av_r:', round(sum_reward[0].item() / num_steps, 3))
            sum_reward = 0.0
            rollout.after_update()
            if episode % model_save_rate == 0:
                model_path = os.path.join(model_save_path, 'ppo_model_' + str(episode) + '.pt')
                torch.save(local_model, model_path)
            if episode % 100 == 0:
                file_path = os.path.join(file_save_path, 'ppo_file.npz')
                np.savez(file_path, loss=np.array(overall_loss), reward=np.array(overall_reward))

    print('training done')
    env.close()
