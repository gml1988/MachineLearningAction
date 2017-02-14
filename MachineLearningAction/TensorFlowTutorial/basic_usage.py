import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

# Use the with expression to control the liftcycle of session variable, will close while going out of scope
with tf.Session() as session:
    result = session.run(product)
    print(result)