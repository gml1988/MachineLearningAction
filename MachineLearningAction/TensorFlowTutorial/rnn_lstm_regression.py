import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20         # The number of time sequence
BATCH_SIZE = 50         # One batch is (time_step * input_size, the number of this data batch
INPUT_SIZE = 1          # For one time, how is the size of the vector to be input, such as an Sin input, the input size will be 1
OUTPUT_SIZE = 1
CELL_SIZE = 10          # Inside the RNN cell, the number of the hidden units
LR = 0.006              # Learning Rate
BATCH_START_TEST = 0

def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50 batches, 20 steps)
    # generate the evenly spaced value in this range [BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    plt.plot(xs[0,:], res[0,:], 'r', xs[0,:], seq[0,:], 'b--')
    plt.show()
    # return the tuple: seq, res and shape (batch, step, input)
    return [seq[:,:,np.newaxis], res[:,:,np.newaxis], xs]

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        # change the data from 3-D to 2-D
        # the new size is (batch * n_step, in_size)
        layer_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        ws_in = self._weight_variable([self.input_size, self.cell_size])
        bs_in = self._bias_variable([self.cell_size,])
        # layer_in_y is (batch * n_step, cell_size)
        with tf.name_scope("wx_plus_b"):
            layer_in_y = tf.matmul(layer_in_x, ws_in) + bs_in
        # reshape layer_in_y ==> (batch, n_steps, cell_size)
        self.layer_in_y = tf.reshape(layer_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self,):
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # assign the 
        with tf.name_scope('initial_state'):
            self.cell_init_state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # because time_major is False, so the matrix shape is [batch_size, max_time ...], the first dimension is not max_time
        self.cell_output, self.cell_final_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=self.layer_in_y, 
            initial_state=self.cell_init_state, 
            time_major=False)

    def add_output_layer(self,):
        # shape = (batch * steps, cell_size)
        layer_out_x = tf.reshape(self.cell_output, [-1, self.cell_size], name='2_2D')
        ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size,])
        # shape = (batch * steps, output_size)
        with tf.name_scope('wx_plus_b'):
            self.pred = tf.matmul(layer_out_x, ws_out) + bs_out

    def compute_cost(self,):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            # Flaten the pred matrix to be 1-D
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name='losses')
        with tf.name_scope('average_cost'):
            # Just one number
            self.cost = tf.div(
                tf.reduce_sum(losses, name="losses_sum"),
                tf.cast(self.batch_size, tf.float32),
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def msr_error(self, y_pre, y_target):
        return tf.square(tf.subtract(y_pre, y_target))

    def _weight_variable(self, shape, name='weights'):
        # get the weight of random normal distribution
        initializer = tf.random_normal_initializer(mean=0, stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        # get the constant bias vector
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    session = tf.Session()
    merged = tf.summary.merge_all()
    # writer = tf.train.summary.FileWriter("logs", session.graph)

    session.run(tf.global_variables_initializer())
    plt.ion()
    for i in range(200):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = session.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred[:TIME_STEPS], 'b-')
        plt.ylim(ymin=-1.2, ymax=1.2)
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))