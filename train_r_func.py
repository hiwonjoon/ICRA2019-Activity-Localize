import sys
import argparse
from pathlib import Path
import random
from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from functools import partial

from nets.rwd_model import OrderBasedRewardFunc, _reacher_arch
from inputs.rwd_dataset import AlignedReacher

def main(args,
         RANDOM_SEED,
         LOG_DIR,
         DATA_TRAJS,
         ALIGNER,
         ALIGNER_KWARGS,
         TARGET_TASK,
         TARGET_SUBTASK,
         BATCH_SIZE,
         TRAIN_NUM,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         SAVE_PERIOD,
         SUMMARY_PERIOD):
    if Path(LOG_DIR).exists():
        print('training seems already done.')
        return

    Path(LOG_DIR).mkdir(parents=True)
    with open(str(Path(LOG_DIR)/'args.txt'),'w') as f:
        f.write( str(args) )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> LOAD DATASET
    if ALIGNER == 'perfect':
        from reacher_eval import PerfectAligner
        aligner = PerfectAligner()
        reacher = AlignedReacher(aligner,
                                DATA_TRAJS,
                                train_ratio=0.8)
    elif ALIGNER == 'maml':
        with tf.Graph().as_default():
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            sess = tf.Session(config=tf_config)

            from reacher_eval import MamlAligner
            aligner = MamlAligner(**eval(ALIGNER_KWARGS))
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess.graph.finalize()
            sess.run(init_op)
            aligner.load(sess)

            reacher = AlignedReacher(aligner,
                                    DATA_TRAJS,
                                    train_ratio=0.8)
            sess.close()
    else:
        assert False
    # <<<<<<<

    with tf.Graph().as_default():
        TARGET_TASK = eval(TARGET_TASK)

        # Specific Target Subtask
        _, (x,y,label)= reacher.build_dataset_shuffle_and_learn_specific(
            TARGET_TASK,
            TARGET_SUBTASK,
            BATCH_SIZE,train=True)
        _, (valid_x,valid_y,valid_label)= reacher.build_dataset_shuffle_and_learn_specific(
            TARGET_TASK,
            TARGET_SUBTASK,
            BATCH_SIZE,train=False)
        # <<<<<<<

        # >>>>>>> MODEL
        with tf.variable_scope('train'):
            # Optimizing
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
            tf.summary.scalar('lr',learning_rate)

            with tf.variable_scope('params') as params:
                pass

            net = OrderBasedRewardFunc(x,y,label,
                        partial(_reacher_arch,32), # embedding vector length
                        learning_rate,
                        0.0, # l2_lambda
                        global_step,
                        params,is_training=True)

        with tf.variable_scope('valid'):
            params.reuse_variables()
            valid_net = OrderBasedRewardFunc(valid_x,valid_y,valid_label,
                                partial(_reacher_arch,32), # embedding vector length
                                None,
                                None, # l2_lambda
                                None,
                                params,is_training=False)

        with tf.variable_scope('misc'):
            # Summary Operations
            tf.summary.scalar('loss',net.loss)
            tf.summary.scalar('acc',net.acc)
            summary_op = tf.summary.merge_all()

            # Initialize op
            init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            #config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

            extended_summary_op = tf.summary.merge([
                tf.summary.scalar('valid_loss',valid_net.loss),
                tf.summary.scalar('valid_acc',valid_net.acc),
            ])

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.graph.finalize()
        sess.run(init_op)

        summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
        #summary_writer.add_summary(config_summary.eval(session=sess))

        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            it,loss,_ = sess.run([global_step,net.loss,net.train_op])

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            if( it % SUMMARY_PERIOD == 0 ):
                tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)

            if( it % (SUMMARY_PERIOD*10) == 0 ): #Extended Summary
                summary = sess.run(extended_summary_op)
                summary_writer.add_summary(summary,it)

        net.save(sess,LOG_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--LOG_DIR',required=True)
    parser.add_argument('--RANDOM_SEED',type=int,default=0)
    parser.add_argument('--DATA_TRAJS',required=True)
    parser.add_argument('--TARGET_TASK',default='(0,1)')
    parser.add_argument('--ALIGNER',default='perfect',choices=['perfect','maml'])
    parser.add_argument('--ALIGNER_KWARGS',default="{'alpha':0.005,'num_sgd':3,'n_way':2,'model_file':'./log/reacher/maml/last.ckpt'}")
    parser.add_argument('--TARGET_SUBTASK',type=int,default=0)

    parser.add_argument('--BATCH_SIZE',type=int,default=32)
    parser.add_argument('--TRAIN_NUM',type=int,default= 60000)
    parser.add_argument('--LEARNING_RATE',type=float,default=0.001)
    parser.add_argument('--DECAY_VAL',type=float,default=1.0)
    parser.add_argument('--DECAY_STEPS',type=int,default=10000)
    parser.add_argument('--DECAY_STAIRCASE',action='store_true')
    parser.add_argument('--SUMMARY_PERIOD',type=int,default=20)
    parser.add_argument('--SAVE_PERIOD',type=int,default=5000)

    args = parser.parse_args()

    main(args = args,**vars(args))
