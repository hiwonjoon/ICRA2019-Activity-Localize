from six.moves import xrange
import tensorflow as tf
from .ops import *

# reacher architecture
def _reacher_arch(num_classes,name='target_specific',linear_embed_size=256):
    net_spec = [
        Conv3d('conv3d_1',3,8,k_t=7,k_h=7,k_w=7,d_h=4,d_w=4),
        BatchNorm('conv3d_1',8,scale=False,axis=-1),
        lambda t,**kwargs : tf.nn.relu(t),
        Conv3d('conv3d_2',8,16,k_t=7,k_h=7,k_w=7,d_h=4,d_w=4),
        BatchNorm('conv3d_2',16,scale=False,axis=-1),
        lambda t,**kwargs : tf.nn.relu(t),
        Conv3d('conv3d_3',16,16,k_t=3,k_h=3,k_w=3,d_h=2,d_w=2),
        BatchNorm('conv3d_3',16,scale=False,axis=-1),
        lambda t,**kwargs : tf.nn.relu(t),
        #lambda t : tf.layers.dropout(t,dropout_rate,training=train),
        Linear('linear_1',16*2*2*16,linear_embed_size),
        lambda t,**kwargs : tf.nn.relu(t),
        #lambda t : tf.layers.dropout(t,dropout_rate,training=train),
        Linear(name,linear_embed_size,num_classes),
    ]
    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights

def _xent_loss(logits,labels,num_classes):
    #Note: sparse version does not have a second derivative..
    #return tf.reduce_mean(
    #    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
    one_hots = tf.one_hot(labels, num_classes, axis=-1)
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hots))

class Maml():
    def __init__(self,alpha,num_sgd,beta,global_step,
                 x_len,x_prime_len,x,y,x_prime,y_prime,
                 arch_fn,loss_fn,
                 param_scope,is_training=False):
        with tf.variable_scope(param_scope):
            net_spec,weights = arch_fn()

        def _per_task(elems):
            x_len,x_prime_len,x,y,x_prime,y_prime = elems
            x_len= tf.cast(x_len,tf.int32)
            x_prime_len= tf.cast(x_prime_len,tf.int32)
            # x, y is [Batch,InputDim], [Batch,OutputDim]
            # return grads of weights per task
            logits_per_steps = []
            task_weights = weights
            for it in range(num_sgd) :
                _t = x
                for block,ws in zip(net_spec,task_weights) :
                    _t = block(_t,is_training=True,**ws)
                task_loss = loss_fn(_t[:x_len],y[:x_len])

                task_weights = [
                    {key: w - alpha*tf.gradients(task_loss,[w])[0] for key,w in ws.items()} #maybe, calculating it as a batch might improve performance..
                    for ws in task_weights
                ]

                _t = x_prime
                for block,ws in zip(net_spec,task_weights) :
                    _t = block(_t,is_training=True,**ws) #update batch norm stats once.
                logits_per_steps.append(_t)

            loss = loss_fn(_t[:x_prime_len],y_prime[:x_prime_len])
            return (logits_per_steps[-1], loss, tf.stack(logits_per_steps,axis=-1))

        with tf.variable_scope('forward') as forward_scope:
            logits, loss, logits_per_steps = tf.map_fn(_per_task,[x_len,x_prime_len,x,y,x_prime,y_prime],dtype=(tf.float32,tf.float32,tf.float32))
            self.logits = logits
            self.loss = tf.reduce_mean(loss)
            self.logits_per_steps = logits_per_steps #shape of logits + [NUM_SGD]


        if( beta is not None ):
            with tf.variable_scope('backward'):
                optimizer = tf.train.AdamOptimizer(beta)
                grads = optimizer.compute_gradients(self.loss)
                self.train_op= optimizer.apply_gradients(grads,global_step=global_step)

                # We will only use on-the-fly statistics for batch norm.
                #with tf.variable_scope('bn_assign') as bn:
                #    print(x)
                #    _t = tf.reshape(x,[-1,1,28,28])
                #    print(_t)
                #    for block,ws in zip(net_spec,weights) :
                #        _t = block(_t,is_training=True,**ws)
                #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,bn.name)
                #print(update_ops)
                #with tf.control_dependencies(update_ops):
                #    grads = optimizer.compute_gradients(self.loss)
                #    self.train_op= optimizer.apply_gradients(grads,global_step=global_step)

        def _sanitize_var_name(var):
            base_scope_name = param_scope.name.split('/')[-1]
            return ('train/'+base_scope_name+'/'.join(var.name.split(base_scope_name)[1:])).split(':')[0]

        save_vars = {_sanitize_var_name(var) : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        #for name,var in save_vars.items():
        #    print(name,var)

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 0)

        sharable_vars = {_sanitize_var_name(var) : var for var in
                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name)
                         if 'target_specific' not in var.name}
        self.sharable_saver = tf.train.Saver(var_list=sharable_vars,max_to_keep = 0)
    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)

    def load_partial(self,sess,model):
        self.sharable_saver.restore(sess,model)


class Classifier():
    def __init__(self,x,y,arch_fn,loss_fn,lr,l2_lambda,global_step,param_scope,is_training=False):
        with tf.variable_scope(param_scope):
            net_spec,weights = arch_fn()

        with tf.variable_scope('forward') as scope:
            _t = x
            blocks = []
            for block in net_spec :
                _t = block(_t,is_training=is_training)
                blocks.append(_t)
            self.feature = blocks[-2]
            self.logits = _t

            self.loss = loss_fn(self.logits,y)
            self.pred = tf.argmax(self.logits,axis=1)
            self.acc = tf.count_nonzero(tf.equal(self.pred,tf.cast(y,tf.int64)),dtype=tf.float64) \
                    / tf.cast(tf.shape(self.logits)[0],tf.float64)

        if (is_training ):
            with tf.variable_scope('backward'):
                self.l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for ws in weights for key,var in ws.items() if key=='w'])

                optimizer = tf.train.AdamOptimizer(lr)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope.name)
                print('batchnorm update ops:',update_ops)
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

    def load(self,sess,model):
        self.saver.restore(sess,model)

