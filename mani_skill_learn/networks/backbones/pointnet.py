import torch
import torch.nn as nn

from mani_skill_learn.utils.data import dict_to_seq
from mani_skill_learn.utils.torch import masked_average, masked_max
from ..builder import BACKBONES, build_backbone

import numpy as np
from numpy import linalg as LA
import os

class PointBackbone(nn.Module):
    def __init__(self):
        super(PointBackbone, self).__init__()

    def forward(self, pcd):
        pcd = pcd.copy()
        if isinstance(pcd, dict):
            if 'pointcloud' in pcd:
                pcd['pcd'] = pcd['pointcloud']
                del pcd['pointcloud']
            assert 'pcd' in pcd
            return self.forward_raw(**pcd)
        else:
            return self.forward_raw(pcd)

    def forward_raw(self, pcd, state=None):
        raise NotImplementedError("")


@BACKBONES.register_module()
class PointNetV0(PointBackbone):
    def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """
        super(PointNetV0, self).__init__()
        conv_cfg = conv_cfg.deepcopy()
        conv_cfg.mlp_spec[0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.global_mlp = build_backbone(mlp_cfg)

    def forward_raw(self, pcd, state, mask=None):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg: shape (l, n_points, n_seg) (unused in this function)
        :param state: shape (l, state_shape) agent state and other information of robot
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """
        if isinstance(pcd, dict):
            pcd = pcd.copy()
            mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
            if self.subtract_mean_coords:
                # Use xyz - mean xyz instead of original xyz
                xyz = pcd['xyz']  # [B, N, 3]
                mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
                pcd['mean_xyz'] = mean_xyz.repeat(1, xyz.shape[1], 1)
                pcd['xyz'] = xyz - mean_xyz
            # Concat all elements like xyz, rgb, seg mask, mean_xyz
            pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
        else:
            mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]

        B, N = pcd.shape[:2]
        state = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, CS]
        point_feature = self.conv_mlp(state.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        global_feature = global_feature.reshape(B, -1)
        return self.global_mlp(global_feature)


@BACKBONES.register_module()
class PointNetWithInstanceInfoV0(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, 
                num_sym_matrix=0, transformer_cfg=None, lstm_cfg=None, matrix_index=-1,
                obj_residual=False):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithInstanceInfoV0, self).__init__()

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg) if final_mlp_cfg is not None else None
        # self.global_lstm = build_backbone(lstm_cfg) if lstm_cfg is not None else None
        if lstm_cfg is not None:
            self.lstm_len = lstm_cfg['lstm_len']
            self.global_lstm = nn.LSTM(lstm_cfg['input_size'], lstm_cfg['hidden_sizes'], num_layers=lstm_cfg['num_layers'])
        else:
            self.lstm_len = 1
            self.global_lstm = None

        self.stack_frame = stack_frame
        self.num_objs = num_objs

        self.obj_residual = obj_residual
        self.num_sym_matrix = num_sym_matrix

        ## random matrix
        # self.eigen_vectors = []
        # dim = final_mlp_cfg['mlp_spec'][0]
        # if self.num_sym_matrix > 0:
        #     for i in range(self.num_sym_matrix):
        #         tmp_matrix = np.random.rand(dim**2).reshape(dim, dim)
        #         tmp_matrix = np.triu(tmp_matrix)
        #         tmp_matrix = tmp_matrix + tmp_matrix.T - np.diag(tmp_matrix.diagonal())
        #         eigen_vector, _ = LA.eigh(tmp_matrix)
        #         eigen_vector = torch.from_numpy(eigen_vector).float()
        #         # eigen_vector.requires_grad = True
        #         # print('eigen_vector:', eigen_vector.shape)
        #         self.eigen_vectors.append(eigen_vector)
            
        #     self.eigen_vectors = torch.stack(self.eigen_vectors, 0)
        #     # self.eigen_vectors = nn.Parameter(self.eigen_vectors, requires_grad=False)
        #     self.eigen_vectors = nn.Parameter(self.eigen_vectors)

        if self.num_sym_matrix > 0:
            dim = final_mlp_cfg['mlp_spec'][0]
            cur_dir = __file__
            for i in range(4):
                # backbones, networks, mani_skill_learn, MANISKILL-LEARN
                cur_dir = os.path.dirname(cur_dir)
            file_name = os.path.join(cur_dir, 'P' + str(num_sym_matrix) + '_matrix_' +str(dim) + '.npy')
            P_matrix = torch.from_numpy(np.load(file_name)).float()
            self.cur_matrix = P_matrix[matrix_index]
            print('matrix_index:', matrix_index)
            print('cur_matrix:', self.cur_matrix.shape)
        else:
            self.cur_matrix = None

        assert self.num_objs > 0

    def forward_raw(self, pcd, state):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg:  shape (l, n_points, n_seg)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """
        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 3]

        B = xyz.shape[0]

        # print('xyz:', xyz.shape)
        # print('rgb:', rgb.shape)
        # print('seg:', seg.shape)
        
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = [] 
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)
        if self.obj_residual:
            handle_feature = obj_features[2].clone()

        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]           
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
            if self.obj_residual:
                global_feature = handle_feature + global_feature
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # print('Y', global_feature.shape)

        if self.global_lstm is not None:
            # obs : [L, batch, input]
            L = self.lstm_len
            if B == 1:
                global_feature = global_feature.view(1, 1, -1)
                global_feature = global_feature.repeat(L, 1, 1)
            else:
                B = B // L 
                # (L, B, -1) wrong!!
                # global_feature = global_feature.view(L, B, -1)
                global_feature = global_feature.view(B, L, -1)
                global_feature = global_feature.permute(1, 0, 2)
            # print('global_feature: ', global_feature.shape)
            outputs, (ht, ct) = self.global_lstm(global_feature)
            # ht is the last hidden state of the sequences
            # ht = (1 x batch_size x hidden_dim)
            # ht[-1] = (batch_size x hidden_dim)
            global_feature = ht[-1]
            # print('after lstm global_feature: ', global_feature.shape)

        # random_global_features = []
        # cur_device = global_feature.get_device()
        # for i in range(len(self.eigen_vectors)):
        #     tmp_eigen_vector = self.eigen_vectors[i].to(cur_device).detach()
        #     tmp = global_feature * tmp_eigen_vector
        #     # print('#########################')
        #     # print('global_feature=%s' % (str(global_feature.shape)))
        #     # print('eigen_vector=%s' % (str(self.eigen_vectors[i].shape)))
        #     # print('tmp=%s' % (str(tmp.shape)))
        #     random_global_features.append(tmp)

        # if len(self.eigen_vectors) > 0:
        #     random_global_features = torch.stack(random_global_features, dim=-1)
        #     global_feature = torch.mean(random_global_features, dim=-1)

        if self.num_sym_matrix > 0 and self.cur_matrix is not None: 
            cur_device = global_feature.get_device()
            self.cur_matrix = self.cur_matrix.to(cur_device)
            # global_feature = global_feature * self.cur_matrix
            global_feature = torch.matmul(global_feature, self.cur_matrix)

        if self.global_mlp is not None:
            x = self.global_mlp(global_feature)
        else:
            x = global_feature
        # print(x)
        return x
