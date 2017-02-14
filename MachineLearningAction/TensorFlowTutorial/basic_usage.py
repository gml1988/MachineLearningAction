import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

# Use the with expression to control the liftcycle of session variable, will close while going out of scope
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
mul = tf.mul(input1, intermed)

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
output = tf.mul(input1, input2)

with tf.Session() as session:
    print(session.run([output], feed_dict = {input1: [7.], input2: [2.]}))