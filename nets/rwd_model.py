from six.moves import xrange
import numpy as np
import tensorflow as tf
from .ops import *

# reacher architecture
def _reacher_arch(num_classes,name='target_specific',linear_embed_size=32):
    net_spec = [
        Conv2d('conv2d_1',3,8,k_h=7,k_w=7,d_h=4,d_w=4,data_format='NHWC'),
        BatchNorm('conv2d_1',8,scale=False,axis=-1),
        lambda t,**kwargs : tf.nn.relu(t),
        Conv2d('conv2d_2',8,16,k_h=7,k_w=7,d_h=4,d_w=4,data_format='NHWC'),
        BatchNorm('conv2d_2',16,scale=False,axis=-1),
        lambda t,**kwargs : tf.nn.relu(t),
        Conv2d('conv2d_3',16,16,k_h=3,k_w=3,d_h=2,d_w=2,data_format='NHWC'),
        BatchNorm('conv2d_3',16,scale=False,axis=-1),
        lambda t,**kwargs : tf.nn.relu(t),
        #lambda t : tf.layers.dropout(t,dropout_rate,training=train),
        Linear('linear_1',2*2*16,linear_embed_size),
        lambda t,**kwargs : tf.nn.relu(t),
        #lambda t : tf.layers.dropout(t,dropout_rate,training=train),
        Linear(name,linear_embed_size,num_classes),
    ]
    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights

def _aligner_arch(linear_embed_size=32):
    net_spec = [
        lambda t,**kwargs : tf.concat(t,axis=1),
        Linear('aligner_linear_1',linear_embed_size*2,linear_embed_size),
        lambda t,**kwargs : tf.nn.relu(t),
        #lambda t : tf.layers.dropout(t,dropout_rate,training=train),
        Linear('classify',linear_embed_size,1),
        lambda t,**kwargs: tf.squeeze(t,axis=1),
    ]
    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights


class OrderBasedRewardFunc():
    # Get To Image Frames -> Output which frame comes first.
    def __init__(self,x,y,label,
                 arch_fn,
                 lr,l2_lambda,global_step,param_scope,is_training=False):
        """
        x: [BATCH_SIZE,H,W,C]
        y: same as x
        label : [BATCH_SIZE] of 0 or 1
        """
        with tf.variable_scope(param_scope):
            net_spec,weights = arch_fn()
            aligner_spec,aligner_weights = _aligner_arch()

            weights += aligner_weights

        with tf.variable_scope('forward') as scope:
            def _get_ev(_t):
                for block in net_spec :
                    _t = block(_t,is_training=is_training)
                return _t

            self.x = x
            self.y = y

            ev_x = _get_ev(x)
            ev_y = _get_ev(y)

            _t = (ev_x,ev_y)
            for block in aligner_spec :
                _t = block(_t,is_training=is_training)
            self.logits = _t
            self.r = _t[1] - _t[0] # Reward Function

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=tf.cast(label,tf.float32)))
            self.pred = tf.greater(tf.sigmoid(self.logits),0.5)
            self.acc = tf.count_nonzero(tf.equal(self.pred,label),dtype=tf.float64) \
                    / tf.cast(tf.shape(self.logits)[0],tf.float64)

        if is_training:
            with tf.variable_scope('backward'):
                self.l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for ws in weights for key,var in ws.items() if key=='w'])

                optimizer = tf.train.AdamOptimizer(lr)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope.name)
                print(update_ops)
                with tf.control_dependencies(update_ops):
                    self.train_op= optimizer.minimize(self.loss + l2_lambda*self.l2_reg,global_step=global_step)

        def _sanitize_var_name(var):
            base_scope_name = param_scope.name.split('/')[-1]
            return ('train/'+base_scope_name+'/'.join(var.name.split(base_scope_name)[1:])).split(':')[0]

        save_vars = {_sanitize_var_name(var) : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 0)

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,graph,sess,model):
        self.graph = graph
        self.sess = sess
        self.saver.restore(sess,model)

    def reward_fn(self,init_img,p_img,c_img):
        with self.graph.as_default():
            r = self.sess.run(self.r,feed_dict={
                self.x: np.repeat(init_img[None],2,axis=0).astype(np.float32)/255.0,
                self.y: np.stack([p_img,c_img],axis=0).astype(np.float32)/255.0,
            })
        return r
