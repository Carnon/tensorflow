import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example

class TrainConfig:
    n_steps = 20
    batch_size = 50
    input_size = 1
    output_size = 1
    cell_size = 10
    lr = 0.006

batch_start = 0

def get_batch(config):
    global batch_start,time_steps
    xs = np.arange(batch_start,batch_start+config.n_steps*config.batch_size).reshape((config.batch_size,config.n_steps))/(10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += config.n_steps

    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

class LSTM(object):
    def __init__(self,config):
        self.n_steps = config.n_steps
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.cell_size = config.cell_size
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.build_RNN()

    def build_RNN(self):
        self.xs = tf.placeholder(tf.float32,[None,self.n_steps,self.input_size])
        self.ys = tf.placeholder(tf.float32,[None,self.n_steps,self.output_size])
        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()
        self.compute_cost()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def add_input_layer(self):
        X_in = tf.reshape(self.xs,[-1,self.input_size])
        #W_in = self._weight_variabel(shape=[self.input_size,self.cell_size])
        #b_in = self._bias_variable(shape=[self.cell_size,])
        W_in = tf.Variable(tf.random_normal(shape=[self.input_size,self.cell_size],dtype=tf.float32))
        b_in = tf.Variable(tf.constant(0.1,tf.float32,[self.cell_size,]))
        Y_in = tf.matmul(X_in,W_in)+b_in
        self.l_in_y = tf.reshape(Y_in,[-1,self.n_steps,self.cell_size])

    def add_cell(self):
        lstm_cell = BasicLSTMCell(self.cell_size)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.cell_outputs,self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False)

    def add_output_layer(self):
        X_out = tf.reshape(self.cell_outputs,[-1,self.cell_size])
        #X_out ==>[batch_size*n_steps ,cell_size]
        #W_out = self._weight_variabel([self.cell_size,self.output_size])
        #b_out = self._bias_variable([self.output_size])
        W_out = tf.Variable(tf.random_normal(shape=[self.cell_size,self.output_size],dtype=tf.float32))
        b_out = tf.Variable(tf.constant(0.1,tf.float32,[self.output_size,]))
        self.pred = tf.matmul(X_out,W_out)+b_out
        #pred ==> [batch_size*n_steps,output_size]

    def compute_cost(self):
        losses = sequence_loss_by_example(
            [tf.reshape(self.pred,[-1])],
            [tf.reshape(self.ys,[-1])],
            [tf.ones([self.batch_size*self.n_steps],dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function= self.msr_error
        )
        self.cost = tf.div(tf.reduce_sum(losses),tf.cast(self.batch_size,tf.float32))

    def msr_error(self,y_pre,y_target):
        return tf.square(tf.subtract(y_pre,y_target))

    def _weight_variabel(self,shape):
        initializer = tf.random_normal_initializer(mean=0,stddev=1.,)
        return tf.get_variable(shape=shape,initializer=initializer)

    def _bias_variable(self,shape):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape,initializer=initializer)

if __name__ == '__main__':
    train_config = TrainConfig()
    model = LSTM(train_config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.show()

    for i in range(200):
        seq,res,xs = get_batch(train_config)
        if i==0:
            feed_dict = {model.xs:seq,model.ys:res}
        else:
            feed_dict = {model.xs:seq,model.ys:res,model.cell_init_state:state}
        _,cost,state,pred = sess.run(
            [model.train_op,model.cost,model.cell_final_state,model.pred],
            feed_dict=feed_dict
        )
        plt.plot(xs[0,:],res[0].flatten(),'r',xs[0,:],pred.flatten()[:train_config.n_steps],'b--')
        plt.ylim(-1.2,1.2)
        plt.draw()
        plt.pause(0.3)
        if i%20 ==0:
            print("cost: ",round(cost,4))

