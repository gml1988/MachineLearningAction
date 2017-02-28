import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()

    def add_input_layer(self,):
        pass
    
    def add_cell(self,):
        pass

    def add_output_layer(self,):
        pass

    def compute_cost(self,):
        pass

if __name__ == '__main__'