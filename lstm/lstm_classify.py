import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data",one_hot=True)

n_input = 28
n_step = 28
n_hidden_units = 128
n_classes = 10
batch_size = 128

x = tf.placeholder(tf.float32,[None,n_step,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

weights = {
    'in':tf.Variable(tf.random_uniform([n_input,n_hidden_units])),
    'out':tf.Variable(tf.random_uniform([n_hidden_units,n_classes]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_classes,]))
}
#X==>[128,28,28]
X = tf.reshape(x,[-1,n_input])
#x==>[128*28,28]
X_in = tf.matmul(X,weights['in'])+biases['in']
#X_in ==>[128*28,128]
X_in = tf.reshape(X_in,[-1,n_step,n_hidden_units])
#X_in ==>[128,28,128]

cell = BasicLSTMCell(n_hidden_units)
init_state = cell.zero_state(batch_size,dtype=tf.float32)
outputs,final_state = tf.nn.dynamic_rnn(cell,X_in,initial_state=init_state)
outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
pred = tf.matmul(outputs[-1],weights['out'])+biases['out']

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < 12800:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_step,n_input])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        if step %20 == 0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step += 1


