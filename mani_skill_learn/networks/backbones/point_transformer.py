import torch
import torch.nn as nn

from mani_skill_learn.utils.data import dict_to_seq
from mani_skill_learn.utils.torch import masked_average, masked_max
from ..builder import BACKBONES, build_backbone

from .point_transformer_utils import index_points, square_distance, PointNetSetAbstraction
import torch.nn.functional as F
import numpy as np

from mani_skill_learn.utils.meta import Registry, build_from_cfg
from mani_skill_learn.utils.torch import masked_average, masked_max

# ATTENTION_LAYERS = Registry('attention layer')

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)

class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        
@BACKBONES.register_module()
class PointTransformerBackbone(nn.Module):
    def __init__(self, num_point, nblocks, nneighbor, transformer_dim, input_dim, fc1_dim):
        super(PointTransformerBackbone, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc1_dim)
        )
        self.transformer1 = TransformerBlock(fc1_dim, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        input_channel = fc1_dim + 3
        # 1200 -> 300 -> 8
        num_point_list = [300, 4]
        for i in range(nblocks):
            channel = fc1_dim * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(num_point_list[i], nneighbor, [input_channel, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
            input_channel = channel + 3
        self.nblocks = nblocks
        
    # def __init__(self, cfg):
    #     super().__init__()
    #     npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
    #     self.fc1 = nn.Sequential(
    #         nn.Linear(d_points, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, 32)
    #     )
    #     self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
    #     self.transition_downs = nn.ModuleList()
    #     self.transformers = nn.ModuleList()
    #     for i in range(nblocks):
    #         channel = 32 * 2 ** (i + 1)
    #         self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
    #         self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
    #     self.nblocks = nblocks
    
    # def forward(self, x):
    #     xyz = x[..., :3]
    #     points = self.transformer1(xyz, self.fc1(x))[0]

    #     xyz_and_feats = [(xyz, points)]
    #     for i in range(self.nblocks):
    #         xyz, points = self.transition_downs[i](xyz, points)
    #         points = self.transformers[i](xyz, points)[0]
    #         xyz_and_feats.append((xyz, points))
    #     return points, xyz_and_feats
    
    def forward(self, x, mask=None):
        xyz = x[..., :3]
        # mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
        # mask = mask[..., None]
        # masked_x = masked_max(x, 1, mask=mask)  # [B, K, CF]
        masked_x = mask * x

        points = self.transformer1(xyz, self.fc1(masked_x))[0]
        print('###### PointTransformerBackbone Called #######')
        print('x=%s' % (str(x.shape)))
        print('xyz=%s' % (str(xyz.shape)))
        print('points=%s' % (str(points.shape)))

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            print("#############################")
            print('i=%s' % (str(i)))
            xyz, points = self.transition_downs[i](xyz, points)
            print('xyz=%s' % (str(xyz.shape)))
            print('points=%s' % (str(points.shape)))
            points = self.transformers[i](xyz, points)[0]
            print('points=%s' % (str(points.shape)))
            xyz_and_feats.append((xyz, points))
        print('###### PointTransformerBackbone Ended #######')
        return points, xyz_and_feats

@BACKBONES.register_module()
class PointTransformerClsV0(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res

@BACKBONES.register_module()
class PointTransformerSegV0(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)

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
class PointTransformerManiV0(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
        """
        Point Transformer with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        only 3 networks (backgound, foreground, all)

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
        super(PointTransformerManiV0, self).__init__()

        # self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        # self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(3)])
        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(1)])
        # # None
        # self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs
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
        rgb = pcd['rgb']  # [B, N, 3]
        B, N = rgb.shape[:2]

        print('xyz:', xyz.shape)
        print('rgb:', rgb.shape)
        print('seg:', seg.shape)

        # obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        # obj_masks = [1. - (torch.sum(seg, dim=-1) < 0.5).type(xyz.dtype)]  # [B, N], the foreground mask
        # for i in range(self.num_objs):
        #     obj_masks.append(seg[..., i])
        # obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud
        obj_masks = [torch.ones_like(seg[..., 0])] # the entire point cloud

        obj_features = [] 
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_mask = obj_mask[..., None]
            print('current i=%d' % (i))
            print('obj_mask=%s' % (str(obj_mask.shape)))
            cur_input = torch.cat((xyz, rgb), dim=-1)
            print('cur_input=%s' % (str(cur_input.shape)))
            cur_input = torch.cat([cur_input, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, xyz+rgb+robot_state]
            print('cur_input=%s' % (str(cur_input.shape)))
            # cur_input[32 1200 44 for drawer] obj_mask [32 1200]
            points_ft, xyz_and_feats = self.pcd_pns[i](cur_input, obj_mask)
            # points_ft [32 4 512]
            print('points_ft=%s' % (str(points_ft.shape)))
            # print('xyz_and_feats=%s' % (str(xyz_and_feats.shape)))
            # xyz = xyz_and_feats[-1][0]
            points_ft = points_ft.view(points_ft.shape[0], -1)
            obj_features.append(points_ft)
            
            # obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)
        # if self.attn is not None:
        #     obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
        #     new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
        #     non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
        #     non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
        #     obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]           
        #     global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        # else:
        #     global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        print('global_feature=%s' % (str(global_feature.shape)))
        # print('Y', global_feature.shape)
        x = self.global_mlp(global_feature)
        print('x=%s' % (str(x.shape)))
        # print(x)
        return x