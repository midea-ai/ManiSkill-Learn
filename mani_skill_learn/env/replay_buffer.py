import glob
import os.path as osp
from random import shuffle

import h5py
import numpy as np

from mani_skill_learn.utils.data import (dict_to_seq, recursive_init_dict_array, map_func_to_dict_array,
                                         store_dict_array_to_h5,
                                         sample_element_in_dict_array, assign_single_element_in_dict_array, is_seq_of)
from mani_skill_learn.utils.fileio import load_h5s_as_list_dict_array, load, check_md5sum
from .builder import REPLAYS


@REPLAYS.register_module()
class ReplayMemory:
    """
    Replay buffer uses dict-array as basic data structure, which can be easily saved as hdf5 file.
    
    See mani_skill_learn/utils/data/dict_array.py for more details.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {}
        self.position = 0
        self.running_count = 0
        self.tracjectory_pos = [0]

    def __getitem__(self, key):
        return self.memory[key]

    def __len__(self):
        return min(self.running_count, self.capacity)

    def reset(self):
        self.memory = {}
        self.position = 0
        self.running_count = 0

    def initialize(self, **kwargs):
        self.memory = recursive_init_dict_array(self.memory, dict(kwargs), self.capacity, self.position)

    def push(self, **kwargs):
        # assert not self.fixed, "Fix replay buffer does not support adding items!"
        self.initialize(**kwargs)
        kwargs = dict(kwargs)
        # kwargs['cnt'] = self.running_count
        # for key, value in kwargs.items():
        #     print("{0} = {1}".format(key, value))
        # kwargs['cnt'] = self.running_count
        assign_single_element_in_dict_array(self.memory, self.position, dict(kwargs))
        self.running_count += 1
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, **kwargs):
        # assert not self.fixed, "Fix replay buffer does not support adding items!"
        kwargs = dict(kwargs)
        keys, values = dict_to_seq(kwargs)
        batch_size = len(list(filter(lambda v: not isinstance(v, dict), values))[0])
        self.tracjectory_pos.append(self.tracjectory_pos[-1] + batch_size)
        for i in range(batch_size):
            self.push(**sample_element_in_dict_array(kwargs, i))

    def sample(self, batch_size, seq_length=1):
        batch_idx = np.random.randint(low=0, high=len(self), size=batch_size)
        if seq_length == 1:
            return sample_element_in_dict_array(self.memory, batch_idx)
        else:
            for i, idx in enumerate(batch_idx):
                for trac_end_idx in self.tracjectory_pos:
                    if idx - trac_end_idx + 1 < seq_length:
                        batch_idx[i] = trac_end_idx + seq_length -1
                        break

                    # if idx >= trac_end_idx:
                    #     continue
                    # else:
                    #     if idx > (trac_end_idx - seq_length):
                    #         batch_idx[i] = trac_end_idx - seq_length
                    #     break

            new_batch_idx = []
            for idx in batch_idx:
                for offset in range(seq_length-1, 0, -1):
                    new_batch_idx.append(idx - offset)
                new_batch_idx.append(idx)
            # print("new_batch_idx: %s", str(new_batch_idx[:15]))
            return sample_element_in_dict_array(self.memory, new_batch_idx)


    def tail_mean(self, num):
        func = lambda _, __, ___: np.mean(_[___ - __:___])
        return map_func_to_dict_array(self.memory, func, num, len(self))

    def get_all(self):
        return sample_element_in_dict_array(self.memory, slice(0, len(self)))

    def to_h5(self, file, with_traj_index=False):
        from h5py import File
        data = self.get_all()
        if with_traj_index:
            data = {'traj_0': data}
        if isinstance(file, str):
            with File(file, 'w') as f:
                store_dict_array_to_h5(data, f)
        else:
            store_dict_array_to_h5(data, file)

    def restore(self, init_buffers, replicate_init_buffer=1, num_trajs_per_demo_file=-1, logger=None):
        buffer_keys = ['obs', 'actions', 'next_obs', 'rewards', 'dones']
        if isinstance(init_buffers, str):
            init_buffers = [init_buffers]
        if is_seq_of(init_buffers, str):
            init_buffers = [load_h5s_as_list_dict_array(_) for _ in init_buffers]
        if isinstance(init_buffers, dict):
            init_buffers = [init_buffers]
        
        if logger is not None:
            logger.info(f'Num of datasets: {len(init_buffers)}')
        print('Num of datasets', len(init_buffers))
        for _ in range(replicate_init_buffer):
            cnt = 0
            for init_buffer in init_buffers:
                for item in init_buffer:
                    # item is a tracjectory
                    if cnt >= num_trajs_per_demo_file and num_trajs_per_demo_file != -1:
                        break
                    item = {key: item[key] for key in buffer_keys}
                    trac_len = len(item['dones'])
                    # for key in item.keys():
                    #     print(key)
                        # print(item[key])
                    tmp = [i for i in range(self.running_count, self.running_count + trac_len)]
                    item['cnt'] = np.array(tmp)
                    # for key in buffer_keys:
                    #     print(key)
                    #     print(item[key])
                    # item['cnt'] = cnt
                    self.push_batch(**item)
                    cnt += 1
                # self.tracjectory_pos.append(cnt)
        
        if logger is not None:
            logger.info(f'Num of buffers {len(init_buffers)}, Total steps {self.running_count}')
        print(f'Num of buffers {len(init_buffers)}, Total steps {self.running_count}')


@REPLAYS.register_module()
class ReplayDisk(ReplayMemory):
    """

    """

    def __init__(self, capacity, keys=None):
        super(ReplayDisk, self).__init__(capacity)
        self.keys = ['obs', 'actions', 'next_obs', 'rewards', 'dones'] if keys is None else keys
        self.h5_files = []
        self.h5_size = []
        self.h5_idx = 0
        self.idx_in_h5 = 0

        self.memory_begin_index = 0

    def restore(self, init_buffers, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
        assert num_trajs_per_demo_file == -1, "For chunked dataset, we only support loading all trajectories"
        assert replicate_init_buffer == 1, "Disk replay does not need to be replicated."
        if not (isinstance(init_buffers, str) and osp.exists(init_buffers) and osp.isdir(init_buffers)):
            print(f'{init_buffers} does not exist or is not a folder!')
            exit(-1)
        else:
            if not osp.exists(osp.join(init_buffers, 'index.pkl')):
                print(f'the index.pkl file should be under {init_buffers}!')
                exit(-1)
            num_files, file_size, file_md5 = load(osp.join(init_buffers, 'index.pkl'))
            h5_files = [osp.abspath(_) for _ in glob.glob(osp.join(init_buffers, '*.h5'))]
            print(f'{num_files} of file in index, {len(h5_files)} files in dataset!')
            if len(h5_files) != num_files:
                print('Wrong index file!')
                exit(0)
            else:
                for name in h5_files:
                    from mani_skill_learn.utils.data import get_one_shape
                    self.h5_files.append(h5py.File(name, 'r'))
                    length = get_one_shape(self.h5_files[-1])[0]
                    index = eval(osp.basename(name).split('.')[0].split('_')[-1])
                    assert file_size[index] == length
                    assert check_md5sum(name, file_md5[index])
                    self.h5_size.append(file_size[index])
        shuffle(self.h5_files)
        self.h5_idx = 0
        self.idx_in_h5 = 0
        self._update_buffer()

    def _get_h5(self):
        if self.idx_in_h5 < self.h5_size[self.h5_idx]:
            return self.h5_files[self.h5_idx]
        elif self.h5_idx < len(self.h5_files) - 1:
            self.h5_idx += 1
        else:
            shuffle(self.h5_files)
            self.h5_idx = 0
        self.idx_in_h5 = 0
        return self.h5_files[self.h5_idx]

    def _update_buffer(self, batch_size=None):
        if self.running_count < self.capacity:
            num_to_add = self.capacity
        elif self.capacity - self.memory_begin_index < batch_size:
            num_to_add = self.memory_begin_index
        else:
            return
        self.memory_begin_index = 0
        while num_to_add > 0:
            h5 = self._get_h5()
            num_item = min(self.h5_size[self.h5_idx] - self.idx_in_h5, num_to_add)
            item = sample_element_in_dict_array(h5, slice(self.idx_in_h5, self.idx_in_h5 + num_item))
            self.push_batch(**item)
            num_to_add -= num_item
            self.idx_in_h5 += num_item
        index = list(range(self.capacity))
        shuffle(index)
        assign_single_element_in_dict_array(self.memory, index, self.memory)

    def sample(self, batch_size):
        assert self.capacity % batch_size == 0
        self._update_buffer(batch_size)
        batch_idx = slice(self.memory_begin_index, self.memory_begin_index + batch_size)
        self.memory_begin_index += batch_size
        return sample_element_in_dict_array(self.memory, batch_idx)
