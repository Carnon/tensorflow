import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class   CnnClassify(object):
    def __init__(self,xs_length=784,ys_length=10):
        self.x_input = tf.placeholder(tf.float32,[None,xs_length],name='input_x')
        self.y_input = tf.placeholder(tf.float32,[None,ys_length],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
        x_images = tf.reshape(self.x_input,[-1,28,28,1])
        #conv1 layer
        W_conv1 = self._weight_variable([5,5,1,32])                     #patch 5*5 ,in size 1, out size 32
        b_conv1 = self._bias_variable([32])
        h_conv1 = tf.nn.relu(self._conv2d(x_images,W_conv1)+b_conv1)    #output size (?,28,28,32)
        h_pool1 = self._max_pool_2x2(h_conv1)                           #output size (?,14,14,32)
        #conv2 layer
        W_conv2 = self._weight_variable([5,5,32,64])                    #pathc 5*5,in size 32,out size 64
        b_conv2 = self._bias_variable([64])
        h_conv2 = tf.nn.relu(self._conv2d(h_pool1,W_conv2)+b_conv2)     #output size (?,14,14,64)
        h_pool2 = self._max_pool_2x2(h_conv2)                           #output size (?,7,7,64)

        #fhuc1 layer
        W_fc1 = self._weight_variable([7*7*64,1024])
        b_fc1 = self._bias_variable([1024])
        #[n_samples,7,7,64] ->> [n_samples,7*7*64]
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1,self.keep_prob)
        #func2 layer
        W_fc2 = self._weight_variable([1024,10])
        b_fc2 = self._bias_variable([10])
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_input*tf.log(prediction),reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

        correct_predict = tf.equal(tf.argmax(prediction,1),tf.argmax(self.y_input,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

    def _weight_variable(self,shape):
        initial = tf.truncated_normal(shape=shape,stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def _conv2d(self,x,W):
        #stride[1,x_movement,y_movement,1]
        #Must stride[0] = stride[3] = 1
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def _max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

if  __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    sess = tf.Session()
    cnn_classify = CnnClassify()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch_xs,batch_ys = mnist.train.next_batch(100)

        if i%5==0:
            print(sess.run(cnn_classify.accuracy,feed_dict={cnn_classify.x_input:batch_xs,
                                                            cnn_classify.y_input:batch_ys,
                                                            cnn_classify.keep_prob:1.0}))
        sess.run(cnn_classify.train_step,feed_dict={cnn_classify.x_input:batch_xs,
                                                    cnn_classify.y_input:batch_ys,
                                                    cnn_classify.keep_prob:1.0})

