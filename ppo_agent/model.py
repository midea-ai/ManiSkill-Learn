import torch
import torch.nn as nn

from mani_skill_learn.utils.data import dict_to_seq
from mani_skill_learn.utils.torch import masked_average, masked_max
from ppo_agent.distributions import DiagGaussian
from mani_skill_learn.networks.builder import POLICYNETWORKS, build_backbone, build_dense_head
from ppo_agent.utils import AddBias, init


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


class PointNetWithInstanceInfoV0(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
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
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs
        assert self.num_objs > 0

    def print_grad(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                # if p.grad is not None:
                #     print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)
                if p.grad is None:
                    print('name: ', name, 'p.requires_grad', p.requires_grad)

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
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0]))  # the entire point cloud

        obj_features = []
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:, :1]), non_empty], dim=-1)  # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # # print('Y', global_feature.shape)
        # x = self.global_mlp(global_feature)
        # # print(x)
        return global_feature


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 64, 16, 30)


class Model(nn.Module):
    def __init__(self, action_dim, latent_dim, sample_mode, nn_cfg, policy_head_cfg, trainable=True):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.backbone = PointNetWithInstanceInfoV0(**nn_cfg)

        self.global_mlp = build_backbone(nn_cfg.final_mlp_cfg)

        self.logstd = AddBias(torch.zeros(action_dim))

        self.policy_header = DiagGaussian()


        policy_head_cfg['scale_prior'] = 1.0
        policy_head_cfg['bias_prior'] = 0.0

        self.expert_policy_head = build_dense_head(policy_head_cfg)

        # critic
        self.critic = nn.Sequential(
            init_(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, 1)),
        )
        if trainable:
            self.train()
        else:
            self.eval()
        self.trainable = trainable
        self.sample_mode = sample_mode

    def act(self, state):
        obs_feature = self.backbone(state)

        value = self.critic(obs_feature)

        action_mean = self.global_mlp(obs_feature)
        action_logstd = self.logstd
        expert_action, _, _, _, _ = self.expert_policy_head(action_mean)
        self.policy_header(action_mean, action_logstd)

        if self.trainable:
            if self.sample_mode == "softmax":
                control = self.policy_header.softmax_sample()
            else:
                control = self.policy_header.sample()
        else:
            control = self.policy_header.sample()

        action_log_probs = self.policy_header.log_probs(control)
        return value.clone().detach(), control.clone().detach(), action_log_probs.clone().detach(), expert_action
        # return value.clone().detach(), expert_action.clone().detach(), action_log_probs.clone().detach(), expert_action

    # def to_device(self, device):
    #     self.backbone.to(device)
    #     self.critic.to(device)
    #     self.control.to_device(device)

    def get_value(self, obs_feature):
        obs_feature = self.backbone(obs_feature)
        value = self.critic(obs_feature)
        return value

    def evaluate_actions(self, state, action):
        """
        obs_feature: [N, *obs_dim]
        action: [N, 3]
        """
        obs_feature = self.backbone(state)

        value = self.critic(obs_feature)
        action_mean = self.global_mlp(obs_feature)
        action_logstd = self.logstd
        self.policy_header(action_mean, action_logstd)

        action_log_probs = self.policy_header.log_probs(action)
        entropy = self.policy_header.entropy()
        return value, action_log_probs, entropy

    def print_grad(self):
        for name, p in self.named_parameters():
            if 'backbone' not in name and 'critic' in name:
                print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)

    def print_parameters(self):
        for name, p in self.named_parameters():
            print('in model---> name: ', name, p.data.mean(), 'p.requires_grad', p.requires_grad)
