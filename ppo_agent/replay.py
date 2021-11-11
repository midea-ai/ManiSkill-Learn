import glob
import h5py
import numpy as np
from mani_skill_learn.utils.data import (dict_to_seq, recursive_init_dict_array, map_func_to_dict_array,
                                         store_dict_array_to_h5,
                                         sample_element_in_dict_array, assign_single_element_in_dict_array, is_seq_of)
from mani_skill_learn.utils.fileio import load_h5s_as_list_dict_array, load, check_md5sum
import random


class ReplayAll():
    def __init__(self, sample_mode, rho, phi, demo_dir, size_buffer, size_buffer_V, init_buffer_size=1):

        """[At the beginning of each trajectory a decision is being made
            from where to sample the next trajectory. The wrapper keeps a buffer of the human
            demonstrations, successful trajectories and trajectories with high maximum value.
            There are as many such buffers as there are workers/environments. The buffers syncronize their
            content periodically]

        Args:
            rho ([Float]): [Probability of sampling a trajectory from the human demonstrations or recorded
            successful trajectories]
            phi ([Float]): [Probability of sampling a trajectory from the value buffer trajectories]
            demo_dir ([string]): [path with the recorded human trajectories]
            size_buffer ([Int]): [maximum size of the the buffer of recorded successful trajectories ]
            size_buffer_V ([Int]): [maximum size of the the value buffer]
        """
        self.rho = rho
        self.phi = phi
        self.phi_ = phi
        self.rho_ = rho
        self.sample_mode = sample_mode
        self.replay = False
        self.demo_dir = demo_dir
        if demo_dir is not None and init_buffer_size >= 1:
            files = glob.glob("{}/*.h5".format(demo_dir))
            self.recordings = self.restore(files, init_buffer_size)
            print('totally has ', len(self.recordings), ' recordings!')
        else:
            self.recordings = []

        self.n_original_demos = len(self.recordings)
        self.size_V_buffer = size_buffer_V
        self.size_R_buffer = size_buffer
        self.size_buffer = size_buffer
        self.recordings_value = []
        self.min_value = -np.inf
        self.value_list_index = None
        self.max_value = 0
        self.mean_value = 0
        self.index_min = None
        self.deleted_index = []

    def restore(self, init_buffers, init_buffer_size, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
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
                    if len(item['rewards']) == 200:
                        print('one demonstration trajectory is discard!')
                        continue
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

    def replay_step(self, action, value):
        '''
        if self.replay is True: return trajectory in replay buffer D_r or D_v
        else: return the input action
        '''
        if self.replay is True:
            # print('num_steps:', self.num_steps, '; step:', self.step)
            if self.num_steps > self.step:
                act = self.acts[self.step]
                obs = self.obs[self.step]
                reward = self.rews[self.step]
                # print(self.value_list_index)
                if self.value_list_index is not None:
                    # print(self.value_list_index, self.deleted_index)
                    if self.value_list_index not in self.deleted_index:
                        self.recordings_value[self.value_list_index]["values"][self.step] = value

                if self.step == (self.num_steps - 1):
                    done = True
                    if self.value_list_index is not None:
                        self.new_max_value = np.max(self.recordings_value[self.value_list_index]["values"])
                        self.max_value_error = self.new_max_value - self.old_max_value
                    else:
                        self.max_value_error = 0.0
                        self.new_max_value = 0.0

                else:
                    done = False
                    if self.value_list_index is not None:
                        if self.value_list_index in self.deleted_index:
                            done = True
                            self.deleted_index = []
                            self.max_value_error = 0.0
                            self.new_max_value = 0.0

                self.step += 1
                return [act, obs, reward, done]

            else:
                return [action]
        else:
            return [action]

    def reset(self):
        rho = self.rho
        phi = self.phi
        max_trajectory_value = np.array([np.max(record['values']) for record
                                         in self.recordings_value])
        self.ps_ = max_trajectory_value
        if len(self.ps_) == 0:
            ps = self.ps_
        else:
            self.max_value = np.max(self.ps_)
            if self.sample_mode == 'value':
                ps = np.abs(np.min(self.ps_)) + self.ps_
            else:
                ps_ = np.array([np.mean(record['rewards']) for record in self.recordings_value])
                ps = np.abs(np.min(ps_)) + ps_

        if len(ps) == 0:
            P = []
        else:
            if np.sum(ps) == 0:
                P = np.ones_like(ps) / len(ps)
            else:
                P = ps * 10 / np.sum(ps * 10)

        if len(self.recordings) != 0:

            if len(self.recordings_value) == 0:
                coin_toss = random.choices([0, 2], weights=[rho, 1 - rho])[0]
            else:
                coin_toss = random.choices([0, 1, 2],
                                           weights=[rho, phi, 1 - phi - rho])[0]

            if coin_toss == 0:
                # choose one trajectory from recording
                # print('choose one trajectory from D_r')
                recording = random.choice(self.recordings)
                self.replay = True
                self.acts = recording['actions']
                self.obs = recording['obs']
                self.rews = recording['rewards']
                self.num_steps = self.acts.shape[0]
                self.step = 0
                self.value_list_index = None

            elif coin_toss == 1:
                # choose one trajectory from D_v
                self.value_list_index = np.random.choice(
                    np.arange(0, len(self.recordings_value)), p=P)
                # print('choose one trajectory from D_v: ', self.value_list_index)

                recording = self.recordings_value[self.value_list_index]
                self.replay = True
                self.acts = recording['actions']
                self.obs = recording['obs']
                self.rews = recording['rewards']

                self.num_steps = self.acts.shape[0]
                self.step = 0
                self.old_max_value = np.max(recording['values'])

            else:
                # don't replay
                # print('No replay, interact with env')
                self.value_list_index = None
                self.replay = False
        else:
            self.replay = False
            self.value_list_index = None

        return self.replay

    def add_demo(self, demo):
        self.recordings.insert(self.n_original_demos, demo)

        if len(self.recordings) > self.size_R_buffer:
            # priority pop previous demo
            # print('pop with FIFO')
            self.recordings.pop()

        if self.size_V_buffer > 0:
            # replay success demo with increasingly probability
            self.rho = self.rho + self.phi_ / self.size_buffer
            # replay failure demo with decreasingly probability
            self.phi = self.phi - self.phi_ / self.size_buffer
            self.size_V_buffer = self.size_V_buffer - 1

        if (len(self.recordings_value) >= self.size_V_buffer) and (len(self.recordings_value) > 0):
            # print('deleting demo from recordings_value')
            if self.sample_mode == 'value':
                max_trajectory_value = np.array([np.max(record['values']) for record in self.recordings_value])
            else:
                max_trajectory_value = np.array([np.mean(record['rewards']) for record in self.recordings_value])

            self.ps_ = max_trajectory_value
            self.min_value = np.min(self.ps_)
            self.index_min = np.argmin(self.ps_)
            # directly pop
            self.recordings_value.pop(self.index_min)
            # todo: debug
            # self.deleted_index.append(self.index_min)
            if self.value_list_index is not None:
                if self.value_list_index > self.index_min:
                    self.value_list_index -= 1
        # print('now there are ', len(self.recordings), len(self.recordings_value))

    def add_demo_value(self, demo):

        if self.size_V_buffer > 0:
            # print(len(self.recordings_value), self.size_V_buffer)
            if len(self.recordings_value) >= self.size_V_buffer:
                # print('replacing demo from recordings_value')
                if self.sample_mode == 'value':
                    max_trajectory_value = np.array([np.max(record['values']) for record in self.recordings_value])
                else:
                    max_trajectory_value = np.array([np.mean(record['rewards']) for record in self.recordings_value])
                self.ps_ = max_trajectory_value
                self.min_value = np.min(self.ps_)
                self.max_value = np.max(self.ps_)
                self.mean_value = np.mean(self.ps_)

                if self.sample_mode == 'value':
                    self.index_min = np.argmin(self.ps_)
                else:
                    ps_ = np.array([np.mean(record['rewards']) for record in self.recordings_value])
                    self.index_min = np.argmin(ps_)

                # directly replace
                self.recordings_value[self.index_min] = demo
                # todo: debug
                # self.deleted_index.append(self.index_min)

            else:
                self.recordings_value.append(demo)
        # print('now there are ', len(self.recordings), len(self.recordings_value))

# ReplayAll(0.1, 0.5, '../example_mani_skill_data/openCabinetDrawer', 10, 10, 'pointcloud')
