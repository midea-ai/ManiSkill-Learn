import glob
import sys
import numpy as np

sys.path.append('/home/quan/ManiSkill-Learn')
from mani_skill_learn.utils.data import (dict_to_seq, recursive_init_dict_array, map_func_to_dict_array,
                                         store_dict_array_to_h5,
                                         sample_element_in_dict_array, assign_single_element_in_dict_array, is_seq_of)
from mani_skill_learn.utils.fileio import load_h5s_as_list_dict_array, load, check_md5sum


def restore(init_buffers, init_buffer_size, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
    buffer_keys = ['obs', 'actions', 'next_obs', 'rewards', 'dones']
    if isinstance(init_buffers, str):
        init_buffers = [init_buffers]
    if is_seq_of(init_buffers, str):
        init_buffers = [load_h5s_as_list_dict_array(_) for _ in init_buffers]
    if isinstance(init_buffers, dict):
        init_buffers = [init_buffers]

    print('Num of datasets', len(init_buffers))
    recordings = []
    enough = False
    for _ in range(replicate_init_buffer):
        cnt = 0
        if enough: break
        for init_buffer in init_buffers:
            if enough: break
            for item in init_buffer:
                if cnt >= num_trajs_per_demo_file and num_trajs_per_demo_file != -1:
                    break
                item = {key: item[key] for key in buffer_keys}
                obs = []
                # if len(item['rewards']) == 200:
                #     print('one demonstration trajectory is discard!')
                #     continue
                for i in range(len(item['rewards'])):
                    data = item['obs']
                    single_obs = {}
                    for keys in data.keys():
                        if isinstance(data[keys], dict):
                            single_obs[keys] = dict()
                            for sub_keys in data[keys]:
                                single_obs[keys][sub_keys] = data[keys][sub_keys][i]
                        else:
                            single_obs[keys] = data[keys][i]
                    obs.append(single_obs)
                item['obs'] = obs
                recordings.append(item)
                if len(recordings) >= init_buffer_size:
                    enough = True
                    break
    return recordings


# demo_dir = '/home/quan/ManiSkill-Learn/demonstrations/drawer'
# files = glob.glob("{}/*.h5".format(demo_dir))
# recordings = restore(files,100)
# data = []
# target = []
# for record in recordings:
#     actions = record['actions']
#     length = actions.shape[0]
#     for i in range(actions.shape[0]):
#         data.append(actions[i])
#         if length == 200:
#             target.append(0)
#         else:
#             target.append(1)
#
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
# tsne.fit_transform(data)
# embeddings = tsne.embedding_
# import matplotlib.pyplot as plt
# target = np.array(target)
# success_data = embeddings[target==1]
# fail_data = embeddings[target==0]
#
# plt.scatter(x=fail_data[:,0], y=fail_data[:,1])
# plt.scatter(x=success_data[:,0], y=success_data[:,1])
# plt.legend(["fail", "success"])
# plt.savefig('action_space_visualization.jpg')


demo_dir = '/home/quan/ManiSkill-Learn/demonstrations/drawer'
files = glob.glob("{}/*.h5".format(demo_dir))
recordings = restore(files, 100)
data = []
target = []
result = []
time_range = 10
for record in recordings:
    actions = record['actions'][:]
    states = record['obs'][:]
    length = actions.shape[0]
    if length == 200:
        continue
    for i in range(actions.shape[0]):
        # data.append(actions[i])
        data.append(list(states[i]['state'])+list(actions[i]))
        target.append(int((i / length) * time_range))

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne.fit_transform(data)
embeddings = tsne.embedding_
import matplotlib.pyplot as plt

target = np.array(target)
legend = []
for i in range(max(target) + 1):
    legend.append(i)
    index = target == i
    new_data = embeddings[index]
    plt.scatter(x=new_data[:, 0], y=new_data[:, 1])

plt.legend(legend)
plt.savefig('action_space_visualization' + str(time_range) + '.jpg')
