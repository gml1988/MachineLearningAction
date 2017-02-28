import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import layer

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
LR = 0.001
TRAINING_ITERS = 100000     
BATCH_SIZE = 128            

N_INPUTS = 28
N_STEPS = 28                # time steps
N_HIDDEN_UNITS = 128        # input 128 images in a batch
N_CLASSES = 10              # MNIST classes (0-9 digits) 

x = tf.placeholder(tf.float32, [None, N_STEPS, N_INPUTS])
y = tf.placeholder(tf.float32, [None, N_CLASSES])

# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([N_INPUTS, N_HIDDEN_UNITS])),
    'out': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[N_HIDDEN_UNITS,])),
    'out': tf.Variable(tf.constant(0.1, shape=[N_CLASSES, ]))
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X (128 batch, 28 steps, 28 inputs)
    # ==> (128*28, 28 inputs)
    X = tf.reshape(X, [-1, N_INPUTS])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    
    X_in = tf.reshape(X_in, [-1, N_STEPS, N_HIDDEN_UNITS])

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0, state_is_tuple=True)
    # LSTM cell is divided into two parts (c_state, m_state)
    _init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(LR).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    step = 0
    while step * BATCH_SIZE < TRAINING_ITERS:
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        batch_xs = batch_xs.reshape([BATCH_SIZE, N_STEPS, N_INPUTS])
        session.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            print(session.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))