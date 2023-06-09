# increase eval_cfg.num_procs for parallel evaluation, preferably 5, 10, or 15
python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=0 \
--work-dir=./test/bc_transformer_door/ \
--resume-from ./full_mani_skill_data/models/OpenCabinetDoor-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDoor-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"

# simple_mani_skill_example
python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=0 --evaluation \
--work-dir=./test/OpenCabinetDrawer_1045_link_0-v0_pcd \
--resume-from=./example_mani_skill_data/OpenCabinetDrawer_1045_link_0-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDrawer_1045_link_0-v0" "eval_cfg.save_video=True" "eval_cfg.num=100" "eval_cfg.num_procs=10" "eval_cfg.use_log=True"

python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=0 \
--work-dir=./test/bc_transformer_drawer/ \
--resume-from ./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"

python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=1 \
--work-dir=./test/bc_transformer_chair/ \
--resume-from ./full_mani_skill_data/models/PushChair-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=PushChair-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"

python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=0 \
--work-dir=./test/bc_transformer_bucket/ \
--resume-from ./full_mani_skill_data/models/MoveBucket-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=MoveBucket-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"
