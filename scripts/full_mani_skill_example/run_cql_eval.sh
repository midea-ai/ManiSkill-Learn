#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'cabinet_models_drawer.yml')))")


# python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/cql_transformer_drawer/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=150000"

# python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --gpu-ids=3 \
# 	--work-dir=./work_dirs/cql_transformer_drawer/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=5000"

python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=3 \
--work-dir=./test/cql_transformer_drawer/ \
--resume-from ./work_dirs/cql_transformer_drawer/CQL/models/model_500.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=10" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"

# python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --gpu-ids=1 \
# 	--work-dir=./work_dirs/cql_transformer_door/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDoor/\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDoor-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=5000"

python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=3 \
--work-dir=./test/cql_transformer_door/ \
--resume-from ./work_dirs/cql_transformer_door/CQL/models/model_500.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDoor-v0" "eval_cfg.num=10" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"

# python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --gpu-ids=2 \
# 	--work-dir=./work_dirs/cql_transformer_chair/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/PushChair/\",\"$model_list\"]" \
# 	"env_cfg.env_name=PushChair-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=5000"

python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=3 \
--work-dir=./test/cql_transformer_chair/ \
--resume-from ./work_dirs/cql_transformer_chair/CQL/models/model_500.ckpt \
--cfg-options "env_cfg.env_name=PushChair-v0" "eval_cfg.num=10" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"

# python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/cql_transformer_bucket/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/MoveBucket/\",\"$model_list\"]" \
# 	"env_cfg.env_name=MoveBucket-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=5000"

python -m tools.run_rl configs/cql/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=3 \
--work-dir=./test/cql_transformer_bucket/ \
--resume-from ./work_dirs/cql_transformer_bucket/CQL/models/model_500.ckpt \
--cfg-options "env_cfg.env_name=MoveBucket-v0" "eval_cfg.num=10" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False"
