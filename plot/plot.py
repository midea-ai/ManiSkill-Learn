import numpy as np
import matplotlib.pyplot as plt
import os

def smooth_reward(reward_list):
    window_length = 10
    windows = []
    smooth_reward_list = []
    for _r in reward_list:
        if len(windows) == window_length:
            windows[0:-1] = windows[1:]
            windows[-1] = _r
        else:
            windows.append(_r)
        smooth_reward_list.append(np.mean(windows))
    smooth_reward_list = np.array(smooth_reward_list)
    return smooth_reward_list


# file_path = '/home/quan/maniskill_model/11-03/20-27-13/files/ppo_file.npz'
# # sample mode == reward
# file_path = '/home/quan/maniskill_model/11-10/23-22-20/files/ppo_file.npz'
# sample mode == value
# file_path = '/home/quan/maniskill_model/11-10/23-27-51/files/ppo_file.npz'
# new sample mode == value
# file_path = '/home/quan/maniskill_model/11-11/21-50-40/files/ppo_file.npz'
# new parameter
# file_path = '/home/quan/maniskill_model/11-12/22-21-44/files/ppo_file.npz'
# file_path = '/home/quan/maniskill_model/11-13/19-23-04/files/ppo_file.npz'
# file_path = '/home/quan/maniskill_model/11-13/22-34-03/files/ppo_file.npz'
CONF = {
    'name_list': ['4 env, no demonstration', '1 env', '4 env'],
    'path': [
        '/home/quan/maniskill_model/11-14/11-11-47',
        '/home/quan/maniskill_model/11-14/21-31-57',
        '/home/quan/maniskill_model/11-13/22-34-03'
    ]
}
for i in range(len(CONF['name_list'])):
    file_path = os.path.join(CONF['path'][i], 'files/ppo_file.npz')
    all_file = np.load(file_path)
    reward = smooth_reward(all_file['reward'])
    plt.plot(reward)
plt.legend(CONF['name_list'])
plt.savefig('ablation_study/reward.jpg')
#
plt.clf()
for i in range(len(CONF['name_list'])):
    file_path = os.path.join(CONF['path'][i], 'files/ppo_file.npz')
    all_file = np.load(file_path)
    ent_loss = all_file['loss'][:, 3]
    plt.plot(ent_loss)
plt.legend(CONF['name_list'])
plt.savefig('ablation_study/ent_loss.jpg')

# losses = all_file['loss']
# sum_total_loss = losses[:, 0]
# sum_action_loss = losses[:, 1]
# sum_value_loss = losses[:, 2]
# sum_ent_loss = losses[:, 3]
#
# plt.clf()
# plt.plot(sum_total_loss)
# plt.savefig('total_loss.jpg')
#
# plt.clf()
# plt.plot(sum_action_loss)
# plt.savefig('action_loss.jpg')
#
# plt.clf()
# plt.plot(sum_value_loss)
# plt.savefig('value_loss.jpg')
#
# plt.clf()
# plt.plot(sum_ent_loss)
# plt.savefig('entropy_loss.jpg')
