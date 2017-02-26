import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import layer

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = session.run(prediction, feed_dict={xs: v_xs})
    # |y_pre| and |v_ys| are both array, argmax is find the index of the max value
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = session.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# The imported data will be 55000 * 10
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = layer.add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
# cross_entropy is loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,
                               mnist.test.labels))