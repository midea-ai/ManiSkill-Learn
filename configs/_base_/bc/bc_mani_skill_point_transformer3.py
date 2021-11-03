log_level = 'INFO'
stack_frame = 1
num_heads = 4

agent = dict(
    type='BC',
    batch_size=64, #128,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='DeterministicHead',
            noise_std=1e-5,
        ),
        nn_cfg=dict(
            type='PointTransformerManiV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            num_sym_matrix=10,
            pcd_pn_cfg=dict(
                type='PointTransformerBackbone',
                nneighbor=16,
                nblocks=4,
                transformer_dim=128,
                num_point=1200,
                input_dim='agent_shape + pcd_xyz_rgb_channel + 3',
                fc1_dim=32,
                num_model=1,
                subtract_mean_coords=True,
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 128, 4 * 128],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                # mlp_spec=['4 * 128 * (num_objs + 3)', 128, 'action_shape * 2'],
                mlp_spec=[4 * 128 * 2, 128, 'action_shape'],
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
