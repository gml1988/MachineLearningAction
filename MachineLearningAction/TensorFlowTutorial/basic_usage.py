import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    session.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # To see the improvement for each step
        if len(ax.lines) != 0:
            ax.lines.remove(ax.lines[0])

        prediction_value = session.run(prediction, feed_dict={xs: x_data})
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.5)
        print("Step {0}: {1}".format(i, session.run(loss, feed_dict={xs: x_data, ys: y_data})))

############################################################
# Plot the result
############################################################


############################################################
# Simple way to use Session
############################################################
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

# Use the "with" expression to control the liftcycle of session variable, will close while going out of scope
with tf.Session() as session:
    result = session.run(product)
    print(result)

############################################################
# Variables: initialize a global variable and update the
# value continously. Check the variable value is shared
# inside the scope.
############################################################
state = tf.Variable(0, name="counter")

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an 'init' Op
# We first have to add the 'init' Op to the graph
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))

    for _ in range(3):
        session.run(update)
        print(session.run(state))

############################################################
# Fetch multiple tensors
############################################################
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as session:
    result = session.run([mul, intermed])
    print(result)

############################################################
# Feed: At first, we need initialied the placeholder,
# the placeholder is the argument for a tensor,
# in the session run phrase, we can pass the value as Feed
# to fill in the placeholder
############################################################
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as session:
    print(session.run([output], feed_dict = {input1: [7.], input2: [2.]}))