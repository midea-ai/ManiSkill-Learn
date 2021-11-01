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
    alpha_optim_cfg=dict(type='Adam', lr=3e-5),
    alpha_prime_optim_cfg=dict(type='Adam', lr=3e-4),
    temperature=1,
    min_q_weight=1,
    min_q_with_entropy=False,
    target_q_with_entropy=True,
    forward_block=2,

    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        nn_cfg=dict(
            type='PointTransformerManiV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointTransformerBackbone',
                nneighbor=16,
                nblocks=4,
                transformer_dim=128,
                num_point=1200,
                input_dim='agent_shape + pcd_xyz_rgb_channel',
                fc1_dim=32,
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 128, 4*128],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                # mlp_spec=['4 * 128 * (num_objs + 3)', 128, 'action_shape * 2'],
                mlp_spec=['4 * 128 * 3', 128, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=3e-4, weight_decay=5e-6),
    ),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointTransformerManiV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointTransformerBackbone',
                nneighbor=16,
                nblocks=4,
                transformer_dim=128,
                num_point=1200,
                input_dim='agent_shape + pcd_xyz_rgb_channel + action_shape',
                fc1_dim=32,
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape', 128, 4*128],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['4 * 128 * 3', 128, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=3e-4, weight_decay=5e-6),
    ),
)

eval_cfg = dict(
    type='Evaluation',
    num=10,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=False,
)

train_mfrl_cfg = dict(
    on_policy=False,
)

env_cfg = dict(
    type='gym',
    unwrapped=False,
    stack_frame=stack_frame,
    obs_mode='pointcloud',
    reward_type='dense',
)
