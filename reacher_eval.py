import argparse
from six.moves import xrange
from six import next
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from functools import partial
import random

from nets.act_model import Classifier, Maml, _reacher_arch, _xent_loss
from inputs.act_dataset import Reacher

FV_SHAPE = [16,64,64,3]

class PerfectAligner():
    def __init__(self):
        pass
    def load(self,sess):
        pass
    def align(self,demo,demo_labels,sample,sample_gt):
        return sample_gt #perfect aligning.

class ClassifierAligner():
    def __init__(self,n_way,model_file):
        fvs = tf.placeholder(tf.float32,[None]+FV_SHAPE)
        labels = tf.placeholder(tf.int32,[None,])
        lr = tf.placeholder(tf.float32,shape=[])

        with tf.variable_scope('classify_aligner'):
            with tf.variable_scope('params') as params:
                pass
            net = Classifier(fvs,labels,
                             partial(_reacher_arch,num_classes=n_way),
                             partial(_xent_loss,num_classes=n_way),
                             None,None,None,
                             params,is_training=False)
            with tf.variable_scope('finetuen'):
                opt = tf.train.GradientDescentOptimizer(lr)
                finetune = opt.minimize(net.loss)

        self.model_file = model_file
        self.fvs = fvs
        self.labels = labels
        self.lr = lr
        self.finetune = finetune
        self.evs = net.feature
        self.net = net

    def load(self,sess):
        self.sess = sess
        self.net.load(sess,self.model_file)

    def fine_tune(self,demos,demos_labels,lr,num_iter=10):
        # Reinitalize parameters.
        pass

    def align(self,demo,demo_labels,sample,sample_gt):
        def _calc_dist(demo_evs,target_evs):
            dist = np.zeros((len(target_evs),len(demo_evs)))
            for i,ev in enumerate(target_evs) :
                dist[i] = np.mean((demo_evs-ev)**2,axis=1) #calculate the distance for every pair of segments
            return dist

        demo_evs = self.sess.run(self.evs,feed_dict={self.fvs:demo,self.labels:demo_labels}) #labels is not used.
        sample_evs = self.sess.run(self.evs,feed_dict={self.fvs:sample,self.labels:sample_gt})

        dists = _calc_dist(demo_evs,sample_evs)
        matches = np.argsort(dists,axis=1)[:,0]
        preds = np.array([demo_labels[idx] for idx in matches])
        return preds

class MamlAligner():
    def __init__(self,alpha,num_sgd,n_way,model_file):
        demo = tf.placeholder(tf.float32,[1,None]+FV_SHAPE)
        demo_labels = tf.placeholder(tf.int64,[1,None])

        sample = tf.placeholder(tf.float32,[1,None]+FV_SHAPE)
        sample_gt = tf.placeholder(tf.int64,[1,None])

        with tf.variable_scope('maml_aligner'):
            with tf.variable_scope('params') as params:
                pass
            net = Maml(alpha,num_sgd,None,None,
                       tf.shape(demo_labels)[1:],tf.shape(sample_gt)[1:],demo,demo_labels,sample,sample_gt,
                       partial(_reacher_arch,num_classes=n_way),
                       partial(_xent_loss,num_classes=n_way),
                       params,is_training=False)

        self.demo = demo
        self.demo_labels = demo_labels
        self.sample = sample
        self.sample_gt = sample_gt
        self.net = net
        self.model_file = model_file

    def load(self,sess):
        self.sess = sess
        self.net.load(sess,self.model_file)

    def align(self,demo,demo_labels,sample,sample_gt):
        # Reindexing labels
        target_labels = np.unique(demo_labels)
        reindexed_labels = np.array([np.where(target_labels==l)[0][0] for l in demo_labels])

        logits = \
            self.sess.run(self.net.logits,feed_dict={self.demo:demo[None],
                                                     self.demo_labels:reindexed_labels[None],
                                                     self.sample:sample[None],
                                                     self.sample_gt:np.zeros((1,len(sample)),np.float32)})
        preds = np.argmax(logits,axis=-1)[0]
        return np.array([target_labels[p] if p < len(target_labels) else -1 for p in preds])

def eval_align(reacher,aligner,eval_per_task=100):
    def _IoU(preds,gt):
        labels = np.unique(np.concatenate([preds,gt],axis=0))
        iou = 0.
        for l in labels :
            inter = np.logical_and(preds==l,gt==l)
            union = np.logical_or(preds==l,gt==l)
            iou += np.count_nonzero(inter)*1. / np.count_nonzero(union)*1.
        return iou / len(labels)

    import itertools
    ious = []
    for task,task_trajs in tqdm(reacher.valid_task_trajs.items()) :
        random.shuffle(task_trajs)
        for i,(x,y) in enumerate(itertools.combinations( range(len(task_trajs)), 2 )):
            demo_fvs,demo_labels = reacher._build_fvs_traj(task_trajs[x][4],task_trajs[x][2])
            sample_fvs,sample_labels = reacher._build_fvs_traj(task_trajs[y][4],task_trajs[y][2])

            preds = aligner.align(demo_fvs,demo_labels,sample_fvs,sample_labels)
            iou = _IoU(preds,sample_labels)
            ious.append(iou)

            if( i >= eval_per_task ):
                break

            #print(sample_labels)
            #print(preds)
            #print(iou)
    return np.mean(ious)

def eval(
    random_seed,
    data_trajs,
    meta_trajs,
    num_frames,
    num_colors,
    num_subtasks,
    maml_model,
    alpha,
    sgd_num,
    classifier_model,
):
    ds = Reacher(data_trajs,num_frames,num_frames//2,allow_mixed=False,train_ratio=0.8)
    meta_ds = Reacher(meta_trajs,num_frames,num_frames//2,allow_mixed=False,train_ratio=0.0)

    maml_aligner = MamlAligner(alpha,sgd_num,num_subtasks,model_file=maml_model)
    class_aligner = ClassifierAligner(num_colors,model_file=classifier_model)

    # Initialize op
    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    maml_aligner.load(sess)
    class_aligner.load(sess)

    for aligner in [maml_aligner,class_aligner]:
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        ious = []
        for _ in range(3):
            iou = eval_align(ds,aligner)
            ious.append(iou)

        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        meta_ious = []
        for _ in range(3):
            meta_iou = eval_align(meta_ds,aligner)
            meta_ious.append(meta_iou)
        print('IoU: %f(%f) / Meta IoU: %f(%f)'%(np.mean(ious),np.std(ious),np.mean(meta_ious),np.std(meta_ious)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--random_seed',default=0)
    parser.add_argument('--data_trajs',required=True)
    parser.add_argument('--meta_trajs',required=True)
    parser.add_argument('--num_frames',default=16)
    parser.add_argument('--num_colors' ,default=4)
    parser.add_argument('--num_subtasks' ,default=2)
    parser.add_argument('--maml_model',required=True)
    parser.add_argument('--alpha',default=0.005)
    parser.add_argument('--sgd_num',default=3)
    parser.add_argument('--classifier_model',required=True)

    args = parser.parse_args()

    eval(**vars(args))
