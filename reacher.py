import argparse
from pathlib import Path
from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from functools import partial
import random

from nets.act_model import Maml, _reacher_arch, _xent_loss
from inputs.act_dataset import Reacher

def main(args,
         RANDOM_SEED,
         PRETRAINED_MODEL,
         LOG_DIR,
         DATA_TRAJS,
         META_TRAJS,
         NUM_FRAMES,
         TASK_NUM,
         N_WAY,
         TRAIN_NUM,
         ALPHA,
         TRAIN_NUM_SGD, #Inner sgd steps.
         VALID_NUM_SGD,
         LEARNING_RATE, #BETA
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

    # >>>>>>> DATASET
    reacher= Reacher(DATA_TRAJS,NUM_FRAMES,NUM_FRAMES//2,allow_mixed=False)
    x_len,x_prime_len,x,y,x_prime,y_prime= reacher.build_queue(TASK_NUM,train=True,num_threads=4)
    x_len_val,x_prime_len_val,x_val,y_val,x_prime_val,y_prime_val= reacher.build_queue(TASK_NUM,train=False,num_threads=1)

    meta_reacher= Reacher(META_TRAJS,NUM_FRAMES,NUM_FRAMES//2,train_ratio=0.0,allow_mixed=False)
    x_len_meta,x_prime_len_meta,x_meta,y_meta,x_prime_meta,y_prime_meta = meta_reacher.build_queue(TASK_NUM,train=False,num_threads=1)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        # Optimizing
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = Maml(ALPHA,TRAIN_NUM_SGD,learning_rate,global_step,x_len,x_prime_len,x,y,x_prime,y_prime,
                   partial(_reacher_arch,num_classes=N_WAY),
                   partial(_xent_loss,num_classes=N_WAY),
                   params,is_training=False)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = Maml(ALPHA,VALID_NUM_SGD,None,None,
                         x_len_val,x_prime_len_val,x_val,y_val,x_prime_val,y_prime_val,
                         partial(_reacher_arch,num_classes=N_WAY),
                         partial(_xent_loss,num_classes=N_WAY),
                         params,is_training=False)
        meta_net = Maml(ALPHA,VALID_NUM_SGD,None,None,
                         x_len_meta,x_prime_len_meta,x_meta,y_meta,x_prime_meta,y_prime_meta,
                         partial(_reacher_arch,num_classes=N_WAY),
                         partial(_xent_loss,num_classes=N_WAY),
                         params,is_training=False)

    with tf.variable_scope('misc'):
        def _get_acc(logits,labels,length):
            #net.logits,labels -> [TASK_NUM,(MAX_NUM_LABELS)*K_SHOTS,N_WAYS]
            tp = tf.cast(tf.equal(tf.argmax(logits,axis=-1),labels),tf.float32)
            mask = tf.sequence_mask(length,dtype=tf.float32)
            acc = tf.reduce_sum(tp*mask,axis=-1) / tf.cast(length,tf.float32)
            return tf.reduce_mean(acc)

        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('acc',_get_acc(net.logits,y_prime,x_prime_len))
        for it in range(TRAIN_NUM_SGD-1):
            tf.summary.scalar('acc_it_%d'%(it),_get_acc(net.logits_per_steps[:,:,:,it],y_prime,x_prime_len))

        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        #config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        extended_summary_op = tf.summary.merge([
            tf.summary.scalar('valid_loss',valid_net.loss),
            tf.summary.scalar('valid_acc',_get_acc(valid_net.logits,y_prime_val,x_prime_len_val)),
            tf.summary.scalar('meta_loss',meta_net.loss),
            tf.summary.scalar('meta_acc',_get_acc(meta_net.logits,y_prime_meta,x_prime_len_meta))] +
            [ tf.summary.scalar('valid_acc_it_%d'%(it),_get_acc(valid_net.logits_per_steps[:,:,:,it],y_prime_val,x_prime_len_val))
             for it in range(VALID_NUM_SGD-1)] +
            [ tf.summary.scalar('meta_acc_it_%d'%(it),_get_acc(meta_net.logits_per_steps[:,:,:,it],y_prime_meta,x_prime_len_meta))
             for it in range(VALID_NUM_SGD-1)])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    if( PRETRAINED_MODEL ):
        net.load_partial(sess,PRETRAINED_MODEL)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    #summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            it,loss,_ = sess.run([global_step,net.loss,net.train_op])
            tqdm.write('[%5d] Loss: %1.3f'%(it,loss))

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            if( it % SUMMARY_PERIOD == 0 ):
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)

            if( it % (SUMMARY_PERIOD*2) == 0 ): #Extended Summary
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
    parser.add_argument('--META_TRAJS',required=True)
    parser.add_argument('--PRETRAINED_MODEL',default=None)
    parser.add_argument('--RANDOM_SEED',type=int,default=0)

    parser.add_argument('--N_WAY',type=int,default=2) #num_subtasks shown in dataset
    parser.add_argument('--TASK_NUM',type=int,default=4) # batch size
    parser.add_argument('--NUM_FRAMES',type=int,default=16)
    parser.add_argument('--TRAIN_NUM',type=int,default=10000) #Size corresponds to one epoch
    parser.add_argument('--ALPHA',type=float,default=0.005)
    parser.add_argument('--TRAIN_NUM_SGD',type=int,default=3)
    parser.add_argument('--VALID_NUM_SGD',type=int,default=3)
    parser.add_argument('--LEARNING_RATE',type=float,default= 0.0001)
    parser.add_argument('--DECAY_VAL',type=float,default=1.0)
    parser.add_argument('--DECAY_STEPS',type=int,default=5000) # Half of the training procedure.
    parser.add_argument('--DECAY_STAIRCASE',action='store_true')
    parser.add_argument('--SUMMARY_PERIOD',type=int,default=20)
    parser.add_argument('--SAVE_PERIOD',type=int,default=1000)

    args = parser.parse_args()

    main(args=args,**vars(args))

    """
    python reacher.py --LOG_DIR ./log/reacher/maml --DATA_TRAJS ./datasets/multi-reacher-easy/given.npz --META_TRAJS ./datasets/multi-reacher-easy/meta.npz --PRETRAINED_MODEL ./log/reacher/classify/model.ckpt-2000
    """
