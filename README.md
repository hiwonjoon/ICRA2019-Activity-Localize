# One-Shot Learning of Multi-Step Tasks from Observation via Activity Localization in Auxiliary Video (ICRA 2019)

Wonjoon Goo and Scott Niekum, University of Texas at Austin

![alt tag](https://github.com/hiwonjoon/ICRA2019-Activity-Localize/raw/master/assets/Figure.png)

This repository contains codes for the ICRA 2019 paper. If you use this code as part of any published research, please consider referring the following paper.

```
@inproceedings{Goo2019,
  title     = {"One-Shot Learning of Multi-Step Tasks from Observation via Activity Localization in Auxiliary Video"},
  author    = {Wonjoon Goo and Scott Niekum},
  year      = {2019},
  booktitle = {2019 IEEE International Conference on Robotics and Automation (ICRA)},
  tppubtype = {inproceedings}
}
```

## Setup Env

The code and following instructions are tested on Ubuntu 16.04.

### Download Code

```
$ git clone https://github.com/hiwonjoon/ICRA2019-Activity-Localize.git --recursive
```

### Prepare Conda Environment

```
$ conda env create -f environment.yml
$ conda activate icra2019_actlocal
$ pip install -r requirements.txt
```

### Prepare Roboschool

```
$ cd roboschool
$ export ROBOSCHOOL_PATH = $(pwd)
$ mkdir bullet3/build
$ cd bullet3/build
$ cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
$ make -j4
$ make install
$ cd ../..
$ pip install -e $ROBOSCHOOL_PATH
```

## Train

1. Prepare Datset

```
$ export DISPLAY:=0
$ python reacher_gen.py
```

2. Train Classifier-based Activity Localizer

```
$ python reacher_classify.py --LOG_DIR ./log/reacher/classify --DATA_TRAJS ./datasets/multi-reacher-easy/given.pkl
```

3. Train MAML-based Activity Localizer

```
$ python reacher.py --LOG_DIR ./log/reacher/maml --DATA_TRAJS ./datasets/multi-reacher-easy/given.pkl --META_TRAJS ./datasets/multi-reacher-easy/meta.pkl --PRETRAINED_MODEL ./log/reacher/classify/model.ckpt-2000
```

4. Train Reward Function

```
$ export DISPLAY:=0
$ python reacher_gen.py --path ./datasets/rwd_learn/multi-reacher-easy --num_trajs_per_task 1000
$ python train_r_func.py --LOG_DIR ./log/reward/perfect --DATA_TRAJ ./datasets/rwd_learn/multi-reacher-easy/given.pkl # Assume perfect localization
$ python train_r_func.py --LOG_DIR ./log/reward/maml --DATA_TRAJ ./datasets/rwd_learn/multi-reacher-easy/given.pkl --ALIGNER 'maml' #maml localization
```

5. Train a policy

```
$ export DISPLAY:=0
$ python train_policy.py --log_dir ./log/policy/maml --reward_type inferred --reward_model ./log/reward/maml/last.ckpt
```
