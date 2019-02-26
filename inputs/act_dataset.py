from six.moves import xrange
import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
import tensorflow as tf
import random
import os,glob

def loadCompressed(fname):
    import gzip, pickle
    with gzip.open(fname) as f:
        obj = pickle.load(f)
    return obj

class Reacher(object):
    def __init__(self,fname,frame_length,step_size,train_ratio=0.8,allow_mixed=False):
        self.frame_length = frame_length
        self.step_size = step_size
        self.allow_mixed = allow_mixed

        f = loadCompressed(fname)
        self.colors = f['colors']
        self.task_trajs = f['trajs']
        self.tasks = [task for task,_ in self.task_trajs.items()]
        self.trajs = [traj for _,trajs in self.task_trajs.items() for traj in trajs]

        self.image_shape = list(self.trajs[0][4].shape[1:])
        print('%d Trajectories loaded'%(len(self.trajs)))

        self.train_task_trajs = {task: trajs[:int(len(trajs)*train_ratio)]
                                 for task,trajs in self.task_trajs.items()}
        self.valid_task_trajs = {task: trajs[int(len(trajs)*train_ratio):]
                                 for task,trajs in self.task_trajs.items()}
        print("Train Trajectories : %d"%(sum([len(item) for key,item in self.train_task_trajs.items()])))
        print("Valid Trajectories : %d"%(sum([len(item) for key,item in self.valid_task_trajs.items()])))

    def stat(self):
        pass

    def _build_fvs_traj(self,frames,gts):
        fvs = []
        labels = []
        for i in xrange(0,len(frames)-self.frame_length,self.step_size):
            if( self.allow_mixed == False and \
                len(np.unique(gts[i:i+self.frame_length])) != 1 ):
                continue #skip this segment cause this segment is not good.
            fvs.append( frames[i:i+self.frame_length] )
            labels.append( gts[i] )
        if( len(fvs) == 0 ):
            return None,None
        return np.stack(fvs, axis=0).astype(np.float32) / 255.0, np.array(labels) #preprocessing features...

    def build_queue(self,task_num,train=True,num_threads=1):
        task_trajs = self.train_task_trajs if train else self.valid_task_trajs
        with tf.device('/cpu'):
            def _get_single_task(np_random):
                task_idx = np_random.choice(len(self.tasks))
                task = self.tasks[task_idx]

                trajs = task_trajs[task]
                i,j = np_random.choice(len(trajs),2,replace=False)
                x,y = \
                    self._build_fvs_traj( trajs[i][4], trajs[i][2] )
                x_prime,y_prime = \
                    self._build_fvs_traj( trajs[j][4], trajs[j][2] )

                # Reindexing labels
                target_labels = list(np_random.permutation(task))
                y = np.array([ target_labels.index(l) for l in y])
                y_prime = np.array([ target_labels.index(l) for l in y_prime])

                x = np.reshape(x,[len(x),-1]) # Reshape for dynamic padded batch
                x_prime = np.reshape(x_prime,[len(x_prime),-1])
                return len(x),len(x_prime),x,y,x_prime,y_prime

            x_len,x_prime_len,x,y,x_prime,y_prime = \
                tf.py_func(partial(_get_single_task,np.random.RandomState(random.randint(0,1000))),
                           [],
                           [tf.int64,tf.int64,tf.float32,tf.int64,tf.float32,tf.int64],
                           stateful=True)
            # Build task batch
            x_len,x_prime_len,x,y,x_prime,y_prime = tf.train.batch(
                [x_len,x_prime_len,x,y,x_prime,y_prime],
                batch_size=task_num,
                num_threads=num_threads,
                capacity=10*task_num,
                dynamic_pad=True,
                shapes=[
                    [],
                    [],
                    [None] + [np.prod([self.frame_length]+self.image_shape)],
                    [None],
                    [None] + [np.prod([self.frame_length]+self.image_shape)],
                    [None],
                ])

            x = tf.reshape(x,[task_num,-1,self.frame_length]+self.image_shape)
            x_prime = tf.reshape(x_prime,[task_num,-1,self.frame_length]+self.image_shape)

            return x_len,x_prime_len,x,y,x_prime,y_prime

    def build_queue_triplet(self,batch_size,train=True,num_threads=1):
        task_trajs = self.train_task_trajs if train else self.valid_task_trajs
        trajs = [ traj for _,trajs in task_trajs.items() for traj in trajs ]

        with tf.device('/cpu'):
            def _get_example(np_random,eq=False) : #TODO: eq=True; guarantees that the prob. of each action will be picked is equal across actions.
                anchor = None
                while(anchor is None) :
                    i = np_random.randint(len(trajs))
                    anchor = self._build_random_fvs(np_random,trajs[i][4],trajs[i][2])

                pos = None
                while (pos is None or anchor[1] != pos[1]):
                    i = np_random.randint(len(trajs))
                    pos = self._build_random_fvs(np_random,trajs[i][4],trajs[i][2], target_label=anchor[1])

                neg = None
                while (neg is None or anchor[1] == neg[1]):
                    i = np_random.randint(len(trajs))
                    neg = self._build_random_fvs(np_random,trajs[i][4],trajs[i][2])

                return anchor[0], pos[0], neg[0], anchor[1], neg[1]

            fv,pos_fv,neg_fv,label,neg_label = \
                tf.py_func(partial(_get_example,np.random.RandomState(random.randint(0,1000))),
                           [],
                           [tf.float32,tf.float32,tf.float32,tf.int64,tf.int64],
                           stateful=True)
            # Build task batch
            return tf.train.batch(
                [fv,pos_fv,neg_fv,label,neg_label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=10*batch_size,
                shapes=[
                    ([self.frame_length]+self.image_shape),
                    ([self.frame_length]+self.image_shape),
                    ([self.frame_length]+self.image_shape),
                    (),
                    (),
                ])

    def _build_random_fvs(self,np_random,frames,gts,target_label=None):
        if( target_label is not None and
            target_label not in np.unique(gts) ):
            return None

        candidates = []
        for i in xrange(0,len(frames)-self.frame_length,self.step_size):
            # skip this segment has two mixed labels
            if( self.allow_mixed == False and \
                len(np.unique(gts[i:i+self.frame_length])) != 1 ):
                continue

            # skip the unwanted segment
            if( target_label is not None and
                gts[i] != target_label ):
                continue

            candidates.append( i )

        if( len(candidates) == 0 ):
            return None

        pts = np_random.choice( candidates )
        return (frames[pts:pts+self.frame_length]).astype(np.float32) / 255.0, gts[pts]
