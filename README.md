# A Minimalist Ensemble Method for Generalizable Offline Deep Reinforcement Learning

This is the offical repository of team Fattonny for the No Interaction Track in [SAPIEN Open-Source Manipulation Skill Challenge](https://sapien.ucsd.edu/challenges/maniskill/2021/).

Our technical report is published on [ICLR 2022 Workshop on Generalizable Policy Learning in Physical World](https://openreview.net/forum?id=BN3b2VpE1Wc).

## Getting Started ##

### Installation ###

Our repository is based on the repository [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn), which is a framework for training agents on [SAPIEN Open-Source Manipulation Skill Challenge](https://sapien.ucsd.edu/challenges/maniskill/2021/), a physics-rich generalizable manipulation skill benchmark over diverse objects with large-scale demonstrations.

ManiSkill-Learn requires the python version to be ```>= 3.6``` and torch version to be ```>= 1.5.0```. We suggest users to use ```python 3.8``` and ```pytorch 1.9.0``` with ```cuda 11.1```. The evaluation system of ManiSkill challenge uses ```python 3.8.10```.

To create our anaconda environment, run

```
cd ManiSkill-Learn-main
conda env create -f environment.yml
```

Then please follow the instructions in [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn#installation) to install [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn#installation) and [ManiSkill](https://github.com/haosulab/ManiSkill).


## Workflow ##

### Training ###

We used the same tool provided in [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn#installation) to train our ensemble models.

To change the permission of the scripts, run
```bash
chmod 777 ./scripts10/full_mani_skill_example/*sh
```

To change the gpu id used, modify the ```--gpu-ids=xx``` in sh.

For drawer tasks, we train 20 models (10_0 -> 10_19) and combine them together to one ensemble model:

```bash
./scripts10/full_mani_skill_example/train_pnt_bc_drawer10_0.sh
./scripts10/full_mani_skill_example/train_pnt_bc_drawer10_1.sh
./scripts10/full_mani_skill_example/train_pnt_bc_drawer10_x.sh
...
./scripts10/full_mani_skill_example/train_pnt_bc_drawer10_19.sh
```

For door tasks, we train 20 models (10_0 -> 10_19) and combine them together to one ensemble model:

```bash
./scripts10/full_mani_skill_example/train_pnt_bc_door10_0.sh
./scripts10/full_mani_skill_example/train_pnt_bc_door10_1.sh
./scripts10/full_mani_skill_example/train_pnt_bc_door10_x.sh
...
./scripts10/full_mani_skill_example/train_pnt_bc_door10_19.sh
```
For chair tasks, we train 20 models (10_0 -> 10_19) and combine them together to one ensemble model:

```bash
./scripts10/full_mani_skill_example/train_pnt_bc_chair10_0.sh
./scripts10/full_mani_skill_example/train_pnt_bc_chair10_1.sh
./scripts10/full_mani_skill_example/train_pnt_bc_chair10_x.sh
...
./scripts10/full_mani_skill_example/train_pnt_bc_chair10_19.sh
```

For bucket tasks, we train 20 models (10_0 -> 10_19) and combine them together to one ensemble model:

```bash
./scripts10/full_mani_skill_example/train_pnt_bc_bucket10_0.sh
./scripts10/full_mani_skill_example/train_pnt_bc_bucket10_1.sh
./scripts10/full_mani_skill_example/train_pnt_bc_bucket10_x.sh
...
./scripts10/full_mani_skill_example/train_pnt_bc_bucket10_19.sh
```

### Download Checkpoints ###

For drawer task: https://drive.google.com/file/d/13JHUN-HJTwRQBng-pKIFOc13nEUFKDeE/view?usp=sharing

For door task: https://drive.google.com/file/d/1KReHkxBKHF-m4rCmQT-bv_I3-jSHpDW8/view?usp=sharing

For chair task: https://drive.google.com/file/d/1UIFKqM54-yEMjIHQER7-PGA6clNiUEOs/view?usp=sharing

For bucket task: https://drive.google.com/file/d/1frUiuV8CHTA0BvED7bw4FTM_P9uspSrC/view?usp=sharing

After downloading the checkpoints, unzip and place them into the folder `./work_dirs/`. 

### Local Evaluation ###
We evaluate the models following [ManiSkill Benchmark Repo](https://github.com/haosulab/ManiSkill#evaluation).
Please follow the steps to run our solution:

- Currently ```user_solution.py``` will run our solution. Please do not change its name.
- Run ```PYTHONPATH=YOUR_SOLUTION_DIRECTORY:$PYTHONPATH python mani_skill/tools/evaluate_policy.py --env ENV_NAME```
  - ```YOUR_SOLUTION_DIRECTORY``` is the directory containing your ```user_solution.py```
  - Specify the levels on which you want to evaluate: ```--level-range 100-200```
  - Note that you should active a python environment supporting your ```user_solution.py``` before running the script
- Result will be exported to ```./eval_results.csv```

## Citation

```
@inproceedings{wu2022minimalist,
  title={A Minimalist Ensemble Method for Generalizable Offline Deep Reinforcement Learning},
  author={Wu, Kun and Zhao, Yinuo and Xu, Zhiyuan and Zhao, Zhen and Ren, Pei and Che, Zhengping and Liu, Chi Harold and Feng, Feifei and Tang, Jian},
  booktitle={ICLR 2022 Workshop on Generalizable Policy Learning in Physical World}
}
```
