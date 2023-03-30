import torch
import torch.nn as nn

from mani_skill_learn.utils.data import dict_to_seq
from mani_skill_learn.utils.torch import masked_average, masked_max
from ..builder import BACKBONES, build_backbone

from .point_transformer_utils import index_points, square_distance, PointNetSetAbstraction
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA

from mani_skill_learn.utils.meta import Registry, build_from_cfg
from mani_skill_learn.utils.torch import masked_average, masked_max

# ATTENTION_LAYERS = Registry('attention layer')

import torch
import torch.nn as nn

# from model.pointnet2.pointnet2_paconv_modules import PointNet2FPModule
# from .pointnet2_paconv_modules import PointNet2FPModule
from mani_skill_learn.paconv_all_util import block
from .pointnet2_paconv_modules import PointNet2SAModuleCUDA as PointNet2SAModule
import os

@BACKBONES.register_module()
class PointNet2SSGSeg(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        c: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, c=3, use_xyz=True, mask_type='*', subtract_mean_coords=True, sa_cfg=None):
        super().__init__()
        sa_cfg_copy = sa_cfg.deepcopy()
        self.nsamples = sa_cfg_copy.get('nsamples', [32, 32, 32, 32])
        self.npoints = sa_cfg_copy.get('npoints', [None, None, None, None])
        self.sa_mlps = sa_cfg_copy.get('sa_mlps', [[c, 32, 32, 64], [64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 256, 512]])
        # self.fp_mlps = config.get('fp_mlps', [[128 + c, 128, 128, 128], [256 + 64, 256, 128], [256 + 128, 256, 256], [512 + 256, 256, 256]])
        self.paconv = sa_cfg_copy.get('pointnet2_paconv', [True, True, True, True, False, False, False, False])
        # self.fc = config.get('fc', 128)
        # self.nsamples = config['nsamples']
        # self.npoints = config['npoints']
        # self.sa_mlps = config['sa_mlps']
        # self.paconv = config['pointnet2_paconv']
        self.mask_type = mask_type
        self.subtract_mean_coords = subtract_mean_coords
        self.c = c

        # if args.get('cuda', False):
        #     from model.pointnet2.pointnet2_paconv_modules import PointNet2SAModuleCUDA as PointNet2SAModule
        # else:
        #     from model.pointnet2.pointnet2_paconv_modules import PointNet2SAModule

        self.SA_modules = nn.ModuleList()
        # print("PointNet2SAModule 1")
        for i in range(len(self.nsamples)):
            self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[i], nsample=self.nsamples[i], mlp=self.sa_mlps[i], use_xyz=use_xyz,
                                                 use_paconv=self.paconv[i], args=sa_cfg_copy))

        # self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[0], nsample=self.nsamples[0], mlp=self.sa_mlps[0], use_xyz=use_xyz,
        #                                          use_paconv=self.paconv[0], args=sa_cfg_copy))
        # # print("PointNet2SAModule 2")
        # self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[1], nsample=self.nsamples[1], mlp=self.sa_mlps[1], use_xyz=use_xyz,
        #                                          use_paconv=self.paconv[1], args=sa_cfg_copy))
        # # print("PointNet2SAModule 3")
        # self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[2], nsample=self.nsamples[2], mlp=self.sa_mlps[2], use_xyz=use_xyz,
        #                                          use_paconv=self.paconv[2], args=sa_cfg_copy))
        # # print("PointNet2SAModule 4")
        # self.SA_modules.append(PointNet2SAModule(npoint=self.npoints[3], nsample=self.nsamples[3], mlp=self.sa_mlps[3], use_xyz=use_xyz,
        #                                          use_paconv=self.paconv[3], args=sa_cfg_copy))
        self.last_conv = nn.Conv1d(self.sa_mlps[-1][-1], self.sa_mlps[-1][-1], 4)
        # self.FP_modules = nn.ModuleList()
        # self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[0], use_paconv=self.paconv[4], args=args))
        # self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[1], use_paconv=self.paconv[5], args=args))
        # self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[2], use_paconv=self.paconv[6], args=args))
        # self.FP_modules.append(PointNet2FPModule(mlp=self.fp_mlps[3], use_paconv=self.paconv[7], args=args))
        # self.FC_layer = nn.Sequential(block.Conv2d(self.fc, self.fc, bn=True), nn.Dropout(), block.Conv2d(self.fc, k, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        # return self.FC_layer(l_features[0])
        # return self.FC_layer(l_features[0].unsqueeze(-1)).squeeze(-1)
        return l_features
    
    def forward_raw(self, pcd, state, mask=None):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg: shape (l, n_points, n_seg) (unused in this function) poped
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
            # Concat all elements like xyz, rgb, mean_xyz
            pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
        else:
            mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]

        B, N = pcd.shape[:2]
        if self.mask_type == '*':
            if self.c == pcd.shape[-1] + state.shape[-1] - 3:
                pcd = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)
            pcd = pcd * mask
        elif self.mask_type == '+':
            if self.c == pcd.shape[-1] + 1 + state.shape[-1] - 3:
                pcd = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)
            pcd = torch.cat([pcd, mask], dim=-1)
        # print('mask:', mask.shape)
        xyz, features = self._break_up_pc(pcd)
        #[256, 1200, 3]
        # print('xyz:', xyz.shape)
        #[256, 44, 1200]
        # print('features:', features.shape)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            # print('l_xyz[i]:', l_xyz[i].shape)
            # print('l_features[i]:', l_features[i].shape)
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            # print('SA i:', i)
            #[256, 300, 3], [256, 75, 3], [256, 18, 3], [256, 4, 3]
            # print('li_xyz:', li_xyz.shape)
            #[256, 64, 300], [256, 128, 75], [256, 256, 18], [256, 512, 4]
            # print('li_features:', li_features.shape)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        # return self.FC_layer(l_features[0])
        # return self.FC_layer(l_features[0].unsqueeze(-1)).squeeze(-1)
        final_feature = self.last_conv(l_features[-1]).squeeze(-1)
        # print('final_feature:', final_feature.shape)
        return final_feature

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
class PAConvPointnet2ManiV0(PointBackbone):
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
        super(PAConvPointnet2ManiV0, self).__init__()

        self.num_objs = num_objs
        # dont use mask of robot
        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)
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

        x = self.global_mlp(global_feature)
        # print(x)
        return x
