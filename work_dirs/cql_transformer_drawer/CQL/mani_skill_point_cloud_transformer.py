log_level = 'INFO'
stack_frame = 1
num_heads = 4
agent = dict(
    type='CQL',
    batch_size=32,
    gamma=0.95,
    update_coeff=0.005,
    target_update_interval=1,
    num_action_sample=8,
    lagrange_thresh=-1,
    alpha=0.2,
    alpha_prime=20,
    automatic_alpha_tuning=True,
    automatic_regularization_tuning=False,
    alpha_optim_cfg=dict(type='Adam', lr=3e-05),
    alpha_prime_optim_cfg=dict(type='Adam', lr=0.0003),
    temperature=1,
    min_q_weight=1,
    min_q_with_entropy=False,
    target_q_with_entropy=True,
    forward_block=2,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead', log_sig_min=-20, log_sig_max=2,
            epsilon=1e-06),
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel', 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=4,
                        latent_dim=32,
                        dropout=0.1),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(
                            type='xavier_init', gain=1, bias=0)),
                    dropout=0.1),
                pooling_cfg=dict(embed_dim=192, num_heads=4, latent_dim=32),
                mlp_cfg=None),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0))),
        optim_cfg=dict(type='Adam', lr=0.0003, weight_decay=5e-06)),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=[
                        'agent_shape + pcd_xyz_rgb_channel + action_shape',
                        192, 192
                    ],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=4,
                        latent_dim=32,
                        dropout=0.1),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(
                            type='xavier_init', gain=1, bias=0)),
                    dropout=0.1),
                pooling_cfg=dict(embed_dim=192, num_heads=4, latent_dim=32),
                mlp_cfg=None),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0))),
        optim_cfg=dict(type='Adam', lr=0.0003, weight_decay=5e-06)))
eval_cfg = dict(
    type='Evaluation',
    num=100,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=False,
    save_video=False,
    use_log=False,
    env_cfg=dict(
        type='gym',
        unwrapped=False,
        stack_frame=1,
        obs_mode='pointcloud',
        reward_type='dense',
        env_name='OpenCabinetDrawer-v0'))
train_mfrl_cfg = dict(
    on_policy=False,
    total_steps=150000,
    warm_steps=0,
    n_steps=0,
    n_updates=500,
    n_eval=5000,
    n_checkpoint=5000,
    init_replay_buffers='',
    init_replay_with_split=[
        './full_mani_skill_data/OpenCabinetDrawer/',
        './ManiSkill/mani_skill/assets/config_files/cabinet_models_drawer.yml'
    ])
env_cfg = dict(
    type='gym',
    unwrapped=False,
    stack_frame=1,
    obs_mode='pointcloud',
    reward_type='dense',
    env_name='OpenCabinetDrawer-v0')
replay_cfg = dict(type='ReplayMemory', capacity=1000000)
work_dir = './work_dirs/cql_transformer_drawer/CQL'
