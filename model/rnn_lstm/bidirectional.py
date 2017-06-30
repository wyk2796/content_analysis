# coding:utf-8

import tensorflow as tf
import numpy as np


class BidirectionalRecurrentN(object):

    def __init__(self):
        self.input_placeholder = None
        self.keep_prob = 1.0
        self.batch_size = None
        self.num_steps = None
        self.embedding = None
        # self.logits = None
        self.cost = 0
        self.train_op = None
        self.predict_op = dict()
        self.accuracy = dict()
        self.outputs = None
        self.learning_rate = 1.0
        self.hidden_size = 200
        # self.output_size = 3
        self.state = 'Train'
        self.variable = dict()
        self.classify_name = []
        self.label_input = dict()
        self.logits = dict()

    def add_input(self):
        self.input_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.batch_size, self.num_steps], name='text_input')


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def set_keep_prob(self, keep_prob):
        self.keep_prob = keep_prob

    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # def set_output_size(self, output_size):
    #     self.output_size = output_size

    def add_embedding(self, vocabulary_size=0, embed_initial=None):
        with tf.variable_scope('word_embedding', reuse=None):
            if embed_initial is None:
                self.embedding = tf.get_variable('embed', shape=[vocabulary_size, self.hidden_size])
            else:
                self.embedding = tf.get_variable('embed', initializer=embed_initial)

    def _lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self.hidden_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)

    def _softmax_weight(self, name, output_size):
        return tf.get_variable(name + '_softmax_weight', [self.hidden_size * 2, output_size], dtype=tf.float32)

    def _softmax_bias(self, name, output_size):
        return tf.get_variable(name + '_softmax_bias', [output_size], dtype=tf.float32)

    def BIRNN(self):

        with tf.device("/cpu:0"):
            input_data = tf.nn.embedding_lookup(self.embedding, self.input_placeholder)
        if self.state == 'Train':
            input_data = tf.nn.dropout(input_data, keep_prob=self.keep_prob)

        input_data = tf.split(tf.reshape(tf.transpose(input_data, [1, 0, 2]), [-1, self.hidden_size]), self.num_steps)
        self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(self._lstm_cell(), self._lstm_cell(),
                                                                     input_data, dtype=tf.float32)

        # weight_s = tf.get_variable('softmax_weight', [self.hidden_size * 2, self.output_size], dtype=tf.float32)
        # bias_s = tf.get_variable('softmax_bias', [self.output_size], dtype=tf.float32)
        # self.logits = tf.matmul(self.outputs[-1], weight_s) + bias_s

    def add_classify(self, name, output_size):
        self.classify_name.append(name)
        self.label_input[name] = tf.placeholder(dtype=tf.int32,
                                                shape=[self.batch_size, output_size], name=name+'_label')
        self.variable[name] = (self._softmax_weight(name, output_size), self._softmax_bias(name, output_size))

    def computer_logits(self):
        for name, (w, b) in self.variable.items():
            self.logits[name] = tf.matmul(self.outputs[-1], w) + b

    def computer_cost(self):
        for (name, logits) in self.logits.items():
            self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=self.label_input[name]))
        self.cost = self.cost / tf.constant(len(self.classify_name), dtype=tf.float32, shape=self.cost.shape)

    def computer_train(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def computer_predict(self):
        for (name, logits) in self.logits.items():
            self.predict_op[name] = tf.argmax(logits, 1, name=name + '_output')

    def computer_accuracy(self):
        for (name, predict_data) in self.predict_op.items():
            correct_pred = tf.equal(predict_data, tf.argmax(self.label_input[name], 1))
            self.accuracy[name] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_train_graph(self, embed_init, vocabulary_size, init_variable, classify):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.state = 'Train'
                [self.add_classify(name, output_size) for (name, output_size) in classify.items()]
                self.add_input()
                self.add_embedding(embed_initial=embed_init, vocabulary_size=vocabulary_size)
                self.BIRNN()
                self.computer_logits()
                self.computer_cost()
                self.computer_train()
                self.computer_predict()
                self.computer_accuracy()

    def get_valid_graph(self, embed_init, vocabulary_size, init_variable, classify):
        with tf.name_scope('Predict'):
            with tf.variable_scope('model', reuse=True, initializer=init_variable):
                self.state = 'Valid'
                [self.add_classify(name, output_size) for (name, output_size) in classify.items()]
                self.add_input()
                self.add_embedding(embed_initial=embed_init, vocabulary_size=vocabulary_size)
                self.BIRNN()
                self.computer_logits()
                self.computer_cost()
                self.computer_predict()
                self.computer_accuracy()

    def get_predict_graph(self, embed_init, vocabulary_size, init_variable, classify):
        with tf.name_scope('Predict'):
            with tf.variable_scope('model', reuse=True, initializer=init_variable):
                self.state = 'Predict'
                [self.add_classify(name, output_size) for (name, output_size) in classify.items()]
                self.keep_prob = 1.0
                self.add_input()
                self.add_embedding(embed_initial=embed_init, vocabulary_size=vocabulary_size)
                self.BIRNN()
                self.computer_logits()
                self.computer_cost()
                self.computer_predict()

    def train_run(self, session, input_data, label_data):
        fetch = {'train_op': self.train_op}
        for (name, accuracy) in self.accuracy.items():
            fetch[name] = accuracy

        feed = {self.input_placeholder: input_data}
        for (name, label) in label_data.items():
            feed[self.label_input[name]] = label

        accuracy = session.run(fetch, feed)
        del accuracy['train_op']
        return accuracy

    def valid_run(self, session, input_data, label_data):
        fetch = dict()
        for (name, accuracy) in self.accuracy.items():
            fetch[name] = accuracy

        feed = {self.input_placeholder: input_data}
        for (name, label) in label_data.items():
            feed[self.label_input[name]] = label

        accuracy = session.run(fetch, feed)
        return accuracy

    def predict_run(self, session, input_data):
        fetch = dict()
        for (name, predict) in self.predict_op.items():
            fetch[name] = predict
        return session.run(fetch, {self.input_placeholder: input_data})




