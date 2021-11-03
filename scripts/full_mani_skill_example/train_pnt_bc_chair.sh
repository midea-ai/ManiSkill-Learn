#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'chair_models.yml')))")
echo ${model_list}

python -m tools.train configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=3 \
	--work-dir=./work_dirs/bc_transformer_chair/ \
	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
	"train_mfrl_cfg.init_replay_with_split=[\"/home/liuchi/zhaoyinuo/ManiSkill-Learn/full_mani_skill_data/pushChair/\",\"$model_list\"]" \
	"env_cfg.env_name=PushChair-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=150000"
