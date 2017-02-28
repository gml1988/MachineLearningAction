import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    # Add one more layer and return the output of the layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    global session
    y_pre = session.run(prediction, feed_dict={xs: v_xs})
    # |y_pre| and |v_ys| are both array, argmax is find the index of the max value
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = session.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
