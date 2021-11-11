python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=0 \
--work-dir=./test/bc_transformer_drawer/ \
--resume-from ./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=3" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False" "eval_cfg.save_video=True"