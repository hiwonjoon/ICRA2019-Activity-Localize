from six.moves import xrange
import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
import tensorflow as tf
import random
import os,glob
import itertools

def loadCompressed(fname):
    import gzip, pickle
    with gzip.open(fname) as f:
        obj = pickle.load(f)
    return obj

class Reacher(object):
    def __init__(self,fname,train_ratio=0.8):
        f = loadCompressed(fname)
        self.colors = f['colors']
        self.task_trajs = f['trajs']
        self.tasks = [task for task,_ in self.task_trajs.items()]
        self.trajs = [traj for _,trajs in self.task_trajs.items() for traj in trajs]

        self.state_shape = list(self.trajs[0][0].shape[1:])
        self.action_shape= list(self.trajs[0][1].shape[1:])
        self.image_shape = list(self.trajs[0][4].shape[1:])
        print('%d Trajectories loaded'%(len(self.trajs)))

        self.train_states, self.train_actions, self.train_videos = \
            zip(*[(states,actions,frames) for states,actions,_,_,frames in self.trajs[:int(len(self.trajs)*train_ratio)]])

        self.valid_states, self.valid_actions, self.valid_videos = \
            zip(*[(states,actions,frames) for states,actions,_,_,frames in self.trajs[int(len(self.trajs)*train_ratio):]])

    def stat(self):
        num_frames = [ len(frames) for _,_,_,_,frames in self.trajs ]
        print( 'Mean #Frames, STD #Frames, Max #Frames, Min #Frames, Total #Frames' )
        print ( np.mean(num_frames), np.std(num_frames), np.max(num_frames), np.min(num_frames), np.sum(num_frames) )

    def build_dataset(self,batch_size,train=True,num_threads=1):
        videos = self.train_videos if(train) else self.valid_videos

        def _read_instance(idx):
            return videos[idx], len(videos[idx])

        ds = tf.data.Dataset.range(len(videos))
        ds = ds.repeat()
        #ds = ds.repeat(1)
        ds = ds.shuffle(len(videos))
        ds = ds.map(lambda idx :
                    tuple(tf.py_func(_read_instance, [idx], [tf.uint8, tf.int64])))
        ds = ds.map(lambda video, seq_len :
                    (tf.cast(tf.reshape(video,[-1]+self.image_shape),tf.float32)/255.0, tf.cast(tf.reshape(seq_len,[]),tf.int32)))
        ds = ds.padded_batch(batch_size,
                             ([-1]+self.image_shape,[]))
        #print(ds.output_types)
        #print(ds.output_shapes)

        #iterator = ds.make_initializable_iterator()
        iterator = ds.make_one_shot_iterator()
        next_batch = iterator.get_next()
        return iterator, next_batch

    def _build_dataset_shuffle_and_learn(self,videos,batch_size):
        def _read_pair(idx):
            v = videos[idx]
            x, y = np.random.choice( len(v) - 30, 2, replace=False )
            return v[x], v[y], x < y

        ds = tf.data.Dataset.range(len(videos))
        ds = ds.repeat()
        #ds = ds.repeat(1)
        ds = ds.shuffle(len(videos))
        ds = ds.map(lambda idx :
                    tuple(tf.py_func(_read_pair, [idx], [tf.uint8, tf.uint8, tf.bool], stateful=True)))
        ds = ds.map(lambda x,y,label :
                    (tf.cast(tf.reshape(x,self.image_shape),tf.float32)/255.0,
                     tf.cast(tf.reshape(y,self.image_shape),tf.float32)/255.0,
                     tf.reshape(label,[])))
        ds = ds.batch(batch_size)
        #print(ds.output_types)
        #print(ds.output_shapes)

        #iterator = ds.make_initializable_iterator()
        iterator = ds.make_one_shot_iterator()
        next_batch = iterator.get_next()
        return iterator, next_batch

    def build_dataset_shuffle_and_learn(self,batch_size,train=True,num_threads=1):
        videos = self.train_videos if(train) else self.valid_videos
        return self._build_dataset_shuffle_and_learn(videos,batch_size)

class AlignedReacher(Reacher):
    #>>>>> Code for aligninig
    def _build_fvs_traj(self,frames,gts,frame_length=16,step_size=8,allow_mixed=False):
        fvs = []
        labels = []
        frame_idxes = [] # Used for referring fvs to frame levels.
        for i in xrange(0,len(frames)-frame_length,step_size):
            if( allow_mixed == False and \
                len(np.unique(gts[i:i+frame_length])) != 1 ):
                continue #skip this segment cause this segment is not good.
            frame_idxes.append( (i,i+frame_length) )
            fvs.append( frames[i:i+frame_length] )
            labels.append( gts[i] )
        if( len(fvs) == 0 ):
            return None,None
        return np.stack(fvs, axis=0).astype(np.float32) / 255.0, np.array(labels), np.array(frame_idxes) #preprocessing features...

    def _pick_frames(self,frames,frame_idxes,pick_locs):
        def _merge(ranges):
            ret = list(ranges[0])
            for st, en in sorted([sorted(t) for t in ranges]):
                if st <= ret[1]:
                    ret[1] = max(ret[1], en)
                else:
                    yield tuple(ret)
                    ret = [st,en]
            yield tuple(ret)
        return np.concatenate([ frames[st:en] for st,en in _merge(frame_idxes[pick_locs])],axis=0)

    def _IoU(self,preds,gt):
        labels = np.unique(np.concatenate([preds,gt],axis=0))
        iou = 0.
        for l in labels :
            inter = np.logical_and(preds==l,gt==l)
            union = np.logical_or(preds==l,gt==l)
            iou += np.count_nonzero(inter)*1. / np.count_nonzero(union)*1.
        return iou / len(labels)
    #<<<<< Code for aligninig

    def __init__(self,aligner,fname,train_ratio=0.8):
        super().__init__(fname,train_ratio=train_ratio)
        self.aligner = aligner

        # Task Aligned Videos
        self.task_aligned_videos = {}
        for task in self.tasks:
            trajs = self.task_trajs[task]

            iou = 0.
            random.shuffle(trajs)
            demo = trajs[0]
            demo_fvs,demo_labels,demo_frame_idxes = self._build_fvs_traj(demo[4],demo[2])

            labels = [(demo_labels,demo_frame_idxes)]
            for sample in trajs[1:]:
                sample_fvs,sample_labels,sample_frame_idxes = self._build_fvs_traj(sample[4],sample[2])
                preds = self.aligner.align(demo_fvs,demo_labels,sample_fvs,sample_labels)

                iou += self._IoU(preds,sample_labels)

                labels.append((preds,sample_frame_idxes))

            print( 'Task %s IOU: %f'%(task,iou/(len(trajs)-1)) )

            assert np.all(np.array(task) == np.unique(demo_labels)), (task, np.unique(demo_labels))
            subtask_videos = {}
            for l in task:
                videos = []
                for (_,_,_,_,frames),(label,frame_idxes) in zip(trajs,labels):
                    if( np.count_nonzero( label == l ) == 0 ):
                        continue

                    picked_frames = self._pick_frames(frames,frame_idxes,np.where(label == l))

                    if( len(picked_frames) <= 30+2):
                        continue
                    videos.append(picked_frames)
                subtask_videos[l] = videos

            self.task_aligned_videos[task] = subtask_videos

        self.train_task_aligned_videos = {
            task: \
              {l: videos[ :int(len(videos)*train_ratio) ]
               for l, videos in subtask_videos.items()}
            for task,subtask_videos in self.task_aligned_videos.items() }
        self.valid_task_aligned_videos = {
            task: \
              {l: videos[ int(len(videos)*train_ratio): ]
               for l, videos in subtask_videos.items()}
            for task,subtask_videos in self.task_aligned_videos.items() }

        # Task Ordered Videos
        self.task_ordered_videos = \
            {order : []
             for task in self.tasks
                for order in itertools.permutations(task) }
        for _,_,gts,_,video in self.trajs:
            ordered_task= tuple([ subtask for subtask,_ in itertools.groupby(gts)])
            self.task_ordered_videos[ordered_task].append(video)
        self.train_task_ordered_videos = \
            {ordered_task: videos[:int(len(videos)*train_ratio)]
             for ordered_task,videos in self.task_ordered_videos.items()}
        self.valid_task_ordered_videos = \
            {ordered_task: videos[int(len(videos)*train_ratio):]
             for ordered_task,videos in self.task_ordered_videos.items()}

        # Task Unordered Videos
        self.task_videos = \
            {task: [traj[4] for traj in trajs]
             for task,trajs in self.task_trajs.items()
            }
        self.train_task_videos = \
            {task: videos[:int(len(videos)*train_ratio)]
             for task,videos in self.task_videos.items()}
        self.valid_task_videos = \
            {task: videos[int(len(videos)*train_ratio):]
             for task,videos in self.task_videos.items()}


    def build_dataset_shuffle_and_learn_whole(self,
                                              task,
                                              batch_size,
                                              train=True,
                                              num_threads=1):
        """
        task: eg) (0,1) or ...
        """
        videos = self.train_task_videos if(train) else self.valid_task_videos
        videos = videos[task]

        return self._build_dataset_shuffle_and_learn(videos,batch_size)


    def build_dataset_shuffle_and_learn_ordered(self,
                                               ordered_task,
                                               batch_size,
                                               train=True,
                                               num_threads=1):
        """
        ordered_task: (0,1) and (1,0) treated differently in this method.
        """
        videos = self.train_task_ordered_videos if(train) else self.valid_task_ordered_videos
        videos = videos[ordered_task]

        return self._build_dataset_shuffle_and_learn(videos,batch_size)

    def build_dataset_shuffle_and_learn_specific(self,
                                                 task,
                                                 target_subtask,
                                                 batch_size,
                                                 train=True,
                                                 num_threads=1):
        """
        task: set of colors used for build a queue. eg) (0,1)
        target_subtask: target for reacher. eg) 0
        """
        videos = self.train_task_aligned_videos if(train) else self.valid_task_aligned_videos
        videos = videos[task][target_subtask]

        return self._build_dataset_shuffle_and_learn(videos,batch_size)
