import os
import sys
import itertools
import argparse
from pathlib import Path
from OpenGL import GLU
import numpy as np
from tqdm import tqdm
import moviepy.editor as mpy
import gym
import roboschool
from roboschool import RoboschoolReacher as RR

def saveCompressed(fname,**objs):
    import gzip, pickle
    with gzip.GzipFile(fname, 'w') as f:
        pickle.dump(objs, f)

def generate(colors,num_subtask,num_trajs_per_task,specific_task=None):
    """
    colors: all the colors used to generate demos. all combination will be generated.
    num_subtask: number of subtasks on each trajectory.
    num_trajs: number of trajectories
    """
    # Set environment
    env = gym.make("RoboschoolReacher-v1")

    # Get policy
    sys.path.append('./roboschool/agent_zoo')
    from RoboschoolReacher_v0_2017may import SmallReactivePolicy
    pi = SmallReactivePolicy(env.observation_space, env.action_space)

    traj_tasks = {}

    if specific_task is not None:
        tasks = [eval(specific_task)]
    else:
        tasks = list(itertools.combinations(range(len(colors)), num_subtask))

    for task in tasks:
        #task: eg) (5,1,2) index of colors
        rest_color_idxes = list(set(range(len(colors)))-set(task))

        trajs = []
        pbar = tqdm(total=num_trajs_per_task)
        while(len(trajs) < num_trajs_per_task):
            env.unwrapped.set_goals( np.random.permutation(range(num_subtask)) )

            sampled_colors = [ colors[idx]
                               for idx in list(task)+list(np.random.choice(rest_color_idxes,4-num_subtask,replace=False)) ]
            env.unwrapped.set_targets_color(sampled_colors)

            states,actions,subtasks,completes,images = [],[],[],[],[]
            obs = env.reset()
            for frames in itertools.count():
                a = pi.act(obs)

                states.append(obs)
                actions.append(a)

                obs, r, done, info = env.step(a)
                img = env.render('rgb_array')

                subtasks.append(task[info['subtask']]) #info['subtask'] is current subtask label
                completes.append(info['complete'])
                images.append(img)

                if done:
                    break

            # Unfinished Demo. Do not save.
            if( frames+1 >= 500 ):
                #tqdm.write('unfinished')
                continue

            # To short actions for subtasks.
            _, counts = np.unique(np.array(subtasks),return_counts=True)
            if( np.min(counts) <= 50 ):
                #tqdm.write('too short subtask')
                continue

            #assert( frames <= 300 )

            # Save
            trajs.append((np.stack(states,axis=0),
                          np.stack(actions,axis=0),
                          np.stack(subtasks,axis=0),
                          np.stack(completes,axis=0),
                          np.stack(images,axis=0)))

            #Update Tqdm Pbar
            pbar.update(1)
        traj_tasks[tuple(task)] = trajs
    return traj_tasks

def gen(args,
        path,
        num_given_colors,
        num_subtasks,
        num_trajs_per_task,
        specific_task,
        video
):
    path = Path(path)
    if path.exists():
        print('data seems already exist.')
        return
    path.mkdir(parents=True)

    if video:
        (path/'video'/'given').mkdir(parents=True)
        (path/'video'/'meta').mkdir(parents=True)

    with open(str(path/'args.txt'),'w') as f:
        f.write( str(args) )

    # Generate Meta-Training Set
    given_colors = RR.COLOR_SET[:num_given_colors]
    given = generate(given_colors, num_subtasks, num_trajs_per_task, specific_task)
    saveCompressed(str(path/'given.pkl'), trajs=given, colors=given_colors )

    if video:
        for task,trajs in given.items():
            for i,(states,actions,subtasks,completes,images) in enumerate(trajs):
                clip = mpy.ImageSequenceClip(list(images),fps=60)
                clip.write_videofile(str(path/'video'/'given'/('%s_%05d.mp4'%(task,i))),
                                     verbose=False,
                                     ffmpeg_params=['-y'],  # always override
                                     progress_bar=False)

    # Generate Meta-Validation Set
    meta_colors = RR.COLOR_SET[num_given_colors:num_given_colors*2]
    meta = generate(meta_colors, num_subtasks, num_trajs_per_task, specific_task)
    saveCompressed(str(path/'meta.pkl'), trajs=meta, colors=meta_colors)

    if video:
        for task,trajs in meta.items():
            for i,(states,actions,subtasks,completes,images) in enumerate(trajs):
                clip = mpy.ImageSequenceClip(list(images),fps=60)
                clip.write_videofile(str(path/'video'/'meta'/('%s_%05d.mp4'%(task,i))),
                                     verbose=False,
                                     ffmpeg_params=['-y'],  # always override
                                     progress_bar=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path',default='./datasets/multi-reacher-easy')
    parser.add_argument('--num_given_colors',type=int,default=4)
    parser.add_argument('--num_subtasks',type=int,default=2)
    parser.add_argument('--num_trajs_per_task',type=int,default=100)
    parser.add_argument('--specific_task',default=None)
    parser.add_argument('--video',action='store_true')

    args = parser.parse_args()

    gen(args=args,**vars(args))

    """
    pytnon reacher_gen.py --path ./datasets/multi-reacher-more-color --num_given_colors 10 --num_subtasks 2 --num_trajs_per_task 40
    pytnon reacher_gen.py --path ./datasets/multi-reacher-three-subtasks --num_given_colors 6 --num_subtasks 3 --num_trajs_per_task 40
    """
