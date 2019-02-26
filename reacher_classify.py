import argparse
from pathlib import Path
from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from functools import partial
import random

from nets.act_model import Classifier, _reacher_arch, _xent_loss
from inputs.act_dataset import Reacher

def main(args,
         RANDOM_SEED,
         LOG_DIR,
         NUM_CLASSES,
         DATA_TRAJS,
         NUM_FRAMES,
         BATCH_SIZE,
         TRAIN_NUM,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         L2_LAMBDA,
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

    # >>>>>>> DATASET
    reacher= Reacher(DATA_TRAJS,NUM_FRAMES,NUM_FRAMES//2,allow_mixed=False)
    fvs, _, _, labels, _ = reacher.build_queue_triplet(BATCH_SIZE,train=True,num_threads=1)
    fvs_valid, _, _, labels_valid, _ = reacher.build_queue_triplet(BATCH_SIZE,train=False,num_threads=1)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        # Optimizing
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass

        net = Classifier(fvs,labels,
                         partial(_reacher_arch,num_classes=NUM_CLASSES),
                         partial(_xent_loss,num_classes=NUM_CLASSES),
                         learning_rate,L2_LAMBDA,global_step,
                         params,is_training=True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = Classifier(fvs_valid,labels_valid,
                               partial(_reacher_arch,num_classes=NUM_CLASSES),
                               partial(_xent_loss,num_classes=NUM_CLASSES),
                               None,None,None,
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
            tf.summary.scalar('valid_acc',valid_net.acc),])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    #summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            it,loss,l2_reg,_,acc = sess.run([global_step,net.loss,net.l2_reg,net.train_op,net.acc])

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            if( it % SUMMARY_PERIOD == 0 ):
                tqdm.write('[%5d] Loss: %1.3f (l2_loss: %1.3f) (acc: %0.3f)'%(it,loss,l2_reg*L2_LAMBDA,acc))
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)

            if( it % (SUMMARY_PERIOD*10) == 0 ): #Extended Summary
                summary = sess.run(extended_summary_op)
                summary_writer.add_summary(summary,it)

    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--LOG_DIR',required=True)
    parser.add_argument('--DATA_TRAJS',required=True)
    parser.add_argument('--RANDOM_SEED',default=0)
    parser.add_argument('--BATCH_SIZE',default=32)
    parser.add_argument('--NUM_CLASSES',default=4) # num_colors shown in dataset
    parser.add_argument('--NUM_FRAMES',default=16)
    parser.add_argument('--TRAIN_NUM',default=10000) #Size corresponds to one epoch
    parser.add_argument('--LEARNING_RATE',default=0.0001)
    parser.add_argument('--DECAY_VAL',default=1.0)
    parser.add_argument('--DECAY_STEPS',default=5000) # Half of the training procedure.
    parser.add_argument('--DECAY_STAIRCASE',default=False)
    parser.add_argument('--L2_LAMBDA',default=0.01)
    parser.add_argument('--SUMMARY_PERIOD',default=10)
    parser.add_argument('--SAVE_PERIOD',default=2000)

    args = parser.parse_args()

    main(args=args,**vars(args))

    """
    python reacher_classify.py --LOG_DIR './log/reacher/classify' --DATA_TRAJS './datasets/multi-reacher-easy/given.npz'
    """
