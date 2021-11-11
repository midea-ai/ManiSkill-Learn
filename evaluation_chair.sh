python -m tools.train configs/bc/mani_skill_point_cloud_transformer.py --evaluation --gpu-ids=0 \
--work-dir=./test/bc_transformer_chair/ \
--resume-from ./submission_example/work_dirs/bc_transformer_chair/BC/models/model_150000.ckpt \
--cfg-options "env_cfg.env_name=PushChair-v0" "eval_cfg.num=3" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False" "eval_cfg.save_video=True"
