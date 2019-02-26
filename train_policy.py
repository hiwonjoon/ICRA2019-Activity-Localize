import os, sys
import argparse
from pathlib import Path
from functools import partial
import numpy as np
import tensorflow as tf
from OpenGL import GLU
import gym, roboschool
from tqdm import tqdm
import moviepy.editor as mpy

from roboschool import RoboschoolReacher as RR
from nets.rwd_model import OrderBasedRewardFunc, _reacher_arch

#TODO: Managing this ugly..sh!
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'baselines'))
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy

def train(env_id, num_timesteps, seed, goal, reward_type, reward_model):
    # Load Reward Function
    if reward_type == 'inferred':
        with tf.Graph().as_default() as g:
            with tf.variable_scope('train'):
                with tf.variable_scope('params') as params:
                    pass

            net = OrderBasedRewardFunc(
                tf.placeholder(tf.float32,[2,64,64,3]),
                tf.placeholder(tf.float32,[2,64,64,3]),
                tf.placeholder(tf.bool,[2]),
                partial(_reacher_arch,32), # embedding vector length
                None,
                None,
                None,
                params,
                is_training=False
            )
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=g,config=config)
        net.load(g,sess,reward_model)

    # Run PPO
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env.unwrapped.set_goals( [goal] )
        env.unwrapped.set_targets_color( RR.COLOR_SET[:4] )

        if reward_type == 'inferred':
            # 4-1. Shuffle And Learn Reward
            env.unwrapped.set_tf_reward_fn(net.reward_fn)

        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    #env = VecNormalize(env,False,False) #normalize observ, normalize ret.

    set_global_seeds(seed)
    ppo2.learn(policy=MlpPolicy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        save_interval=10)

def test(env_id, seed, log_dir, model, goal=0, num_iter=100, video_path=None):
    model_path = str(Path(log_dir)/'checkpoints'/('%05d'%model))

    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)

        def make_env():
            env = gym.make(env_id)
            env.unwrapped.set_goals( [goal] )
            env.unwrapped.set_targets_color( RR.COLOR_SET[:4] )
            return env

        set_global_seeds(seed)

        # Make Env
        env = DummyVecEnv([make_env],render=False)
        env = VecNormalize(env,True,True)
        env.load(model_path)

        # Make Model
        model = ppo2.Model(policy=MlpPolicy, ob_space=env.observation_space, ac_space=env.action_space,
                           nbatch_act=1, nbatch_train=1,nsteps=1,ent_coef=0.0,vf_coef=0.5,max_grad_norm=0.5)
        model.load(model_path)

        def _gen_traj():
            obs = env.reset()
            states,actions,images,done = [obs[0]], [], [], [False]
            while not done[0]:
                a, _, _, _ = model.step(obs,model.initial_state,done)
                obs, r, done, info = env.step(a)

                states.append(obs[0])
                actions.append(a[0])
                images.append(info[0]['img'])

            return states, actions, images

        if video_path is not None:
            video_path = Path(video_path)
            video_path.mkdir(parents=True,exist_ok=True)

        success = 0
        for it in tqdm(range(num_iter)):
            states, actions, images = _gen_traj()

            if( len(states) < 150):
                success+=1

            if video_path is not None:
                clip = mpy.ImageSequenceClip(list(images),fps=60)
                clip.write_videofile(str(video_path/('video_%d.mp4'%it)),verbose=False,ffmpeg_params=['-y'],progress_bar=False)

        print('Success Rate: %f(%d/%d)'%(1.*success/num_iter,success,num_iter))
        sess.close()
    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--env', help='environment ID', default='RoboschoolReacher-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--goal', default=0, choices=[0,1])
    parser.add_argument('--reward_type', default='inferred', choices=['gt','inferred'])
    parser.add_argument('--reward_model', default='')
    # For Eval
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--model',type=int,default=480)
    parser.add_argument('--num_iter',type=int,default=100)
    parser.add_argument('--video_path',default=None)
    args = parser.parse_args()

    if not args.eval:
        logger.configure(args.log_dir)
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, goal=args.goal, reward_type=args.reward_type, reward_model=args.reward_model)
    else:
        test(args.env, args.seed, args.log_dir, args.model, args.goal, args.num_iter, args.video_path)
