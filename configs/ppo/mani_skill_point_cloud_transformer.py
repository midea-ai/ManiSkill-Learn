_base_ = ['../_base_/bc/bc_mani_skill_pointnet_transformer.py']

# env_cfg = dict(
#     type='gym',
#     env_name='OpenCabinetDrawer-v0',
# )

env_cfg = dict(
    type='gym',
    unwrapped=False,
    obs_mode='pointcloud',
    # reward_type='dense',
    reward_type='self_design',
    # env_name='OpenCabinetDrawer-v0',
    env_name='OpenCabinetDrawer_1045-v0',
)

replayer_cfg = dict(
    sample_mode = "value",
    # sample_mode="reward",

    # threshold_reward = 20,
    # threshold_reward=-2.3,
    threshold_reward=-1,
    # rho=0.5,
    # rho=0.0,
    # phi=0.5,
    rho=0.1,
    phi=0.3,
    # rho=0.0,
    # phi=0.0,
    init_buffer_size=10,
    # size_buffer=3,
    # size_buffer_V=3,
    size_buffer=40,
    size_buffer_V=40,
    # demo_dir='example_mani_skill_data/openCabinetDrawer',
    demo_dir='demonstrations/drawer',
)

rollout_cfg = dict(
    num_steps=256,
    num_process=4,
    # num_process=1,
    # num_steps=32,
    mini_batch_num=4,
    use_gae=True,
    gamma=0.99,
    gae_param=0.95
)

model_cfg = dict(
    sample_mode='sample',
    latent_dim=256
)

train_mfrl_cfg = dict(
    total_episode=3000,
    # total_episode=3,
    use_adv_norm=True,
    # use_adv_norm=False,
    warm_up=True,
    # warm_up=False,
    ppo_epoch=4,
    clip=0.1,
    value_coeff=0.01,
    action_coeff=10.0,
    ent_coeff=0.1,
    lr=3e-4,
    model_save_rate=500,
    # model_save_rate=1,
    # device=0,
    device=1,
    # root_path='/home/quan/maniskill_model',
    root_path='/home/liuchi/zhaoyinuo/maniskill_model',
    env_mode='pointcloud',
    expert_cfg_file='configs/bc/mani_skill_point_cloud_transformer.py',
    # expert_cfg_model='/home/quan/ManiSkill-Learn/full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt',
    expert_cfg_model='/home/liuchi/zhaoyinuo/ManiSkill-Learn/full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt',

)

eval_cfg = dict(
    num=10,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=False,
    use_log=False,
)
