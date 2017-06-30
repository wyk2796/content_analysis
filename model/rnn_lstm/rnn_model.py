# coding:utf-8

import tensorflow as tf
import model.train_data_generate as dg
import model.rnn_lstm.rnn_config as conf
import time
import numpy as np

class RNNModel(object):

    def __init__(self, config, operation):
        self.config = config
        self.operation = operation
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='RNN/Input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='RNN/Target')
        self.cell = None
        self.embeddings = None
        self._cost = None
        self._initial_state = None
        self._final_state = None
        self.logits = None
        self._train_op = None
        self._lr = None
        self._new_lr = None
        self.predict = None
        self.forward_computer()
        if operation == 'Train':
            self.computer_cost()
            self.computer_train()
        if operation == 'Valid':
            self.computer_cost()
        if operation == 'Predict':
            self.computer_predict()
        if operation == 'Predict_argmax':
            self.computer_argmax_predict()

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self.config.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

    def create_cell(self):
        attn_cell = self.lstm_cell
        if self.operation == 'Train' and self.config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                   self.lstm_cell(), output_keep_prob=self.config.keep_prob)
        self.cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
        self._initial_state = self.cell.zero_state(self.config.batch_size, dtype=tf.float32)

    def get_embedding(self):
        self.embeddings = tf.get_variable("embedding",
                                          initializer=self.config.embedding_init,
                                          dtype=tf.float32)

    def forward_computer(self):
        self.create_cell()
        with tf.device("/cpu:0"):
            self.get_embedding()
            inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        if self.operation == 'Train' and self.config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, self.config.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [self.config.hidden_size, self.config.vocabulary_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.config.vocabulary_size], dtype=tf.float32)
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self._final_state = state

    def computer_train(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def computer_cost(self):
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.labels_placeholder, [-1])],
            [tf.ones([self.config.batch_size * self.config.num_steps], dtype=tf.float32)])
        self._cost = tf.reduce_sum(loss) / self.config.batch_size

    def computer_predict(self):
        self.predict = tf.nn.softmax(logits=self.logits)

    def computer_argmax_predict(self):
        self.predict = tf.argmax(tf.nn.softmax(logits=self.logits), axis=1)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def load_embedding_from_word2vec(self, session, word2vec, path):
        try:
            word2vec.load_model(session, path)
            self.embeddings = word2vec.get_embedding()
        except Exception as e:
            print('load embedding encounter an error!', e)

    def train(self, session, data_generator):
        start_time = time.time()
        iters = 0
        costs = 0
        epoch_size = ((data_generator.size // self.config.batch_size) - 1) // self.config.num_steps
        state = session.run(self._initial_state)
        fetch = {
            'train_op': self._train_op,
            "cost": self._cost,
            "final_state": self._final_state
        }
        for step, (train_input, train_label) in enumerate(data_generator.generate_batch()):
            feed = {self.input_placeholder: train_input,
                    self.labels_placeholder: train_label,
                    self._initial_state: state}
            vals = session.run(
                fetch, feed_dict=feed)
            costs += vals['cost']
            state = vals['final_state']
            iters += self.config.num_steps
            if step % (epoch_size // 10) == 0:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters),
                       iters * self.config.batch_size / (time.time() - start_time)))
        return np.exp(costs / iters)

    def train_td(self, session, train_data):
        start_time = time.time()
        iters = 0
        costs = 0
        epoch_size = train_data.size() // self.config.batch_size
        state = session.run(self._initial_state)
        fetch = {
            'train_op': self._train_op,
            "cost": self._cost,
            "final_state": self._final_state
        }
        for step, (train_input, train_label) in enumerate(train_data.train_data_content(self.config.batch_size,
                                                                                        self.config.num_steps)):
            feed = {self.input_placeholder: train_input,
                    self.labels_placeholder: train_label,
                    self._initial_state: state}
            vals = session.run(
                fetch, feed_dict=feed)
            costs += vals['cost']
            state = vals['final_state']
            iters += self.config.num_steps
            if step % (epoch_size // 10) == 0:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters),
                       iters * self.config.batch_size / (time.time() - start_time)))
        return np.exp(costs / iters)

    def run_predict(self, session, data, stop_num=100):
        state = session.run(self._initial_state)
        predict_data = data
        print(data)
        for j in range(stop_num):
            fetches = {
                "predict": self.predict,
                "final_state": self._final_state,
            }
            feed_dict = {
                self.input_placeholder: [predict_data[-1:]],
                self._initial_state: state
            }
            vals = session.run(fetches, feed_dict)
            y_predict = vals["predict"][-1]
            new_ids = self.sample(y_predict, 1.0)
            predict_data.append(new_ids)
            state = vals["final_state"]
            # self.batch_size = 1
            # self.num_steps = len(predict_data)
        print(predict_data)
        return predict_data

    def run_predict_td(self, session, predict_data, wc):
        state = session.run(self._initial_state)
        fetches = {
            "predict": self.predict,
            "final_state": self._final_state,
        }
        predict = []
        for step, (train_input, label) in enumerate(predict_data.train_data_content(self.config.batch_size,
                                                                                    self.config.num_steps)):
            feed = {self.input_placeholder: train_input,
                    self._initial_state: state}
            vals = session.run(
                fetches, feed_dict=feed)
            y_predict = [w for w in vals["predict"] if w != 1 and w != 0]
            labelw = [w for w in label[0] if w != 1]
            trainw = [w for w in train_input[0] if w != 1]
            state = vals["final_state"]


            predict.append(y_predict)
            print('y_predict', wc.ids_to_words(y_predict), wc.ids_to_words(labelw), wc.ids_to_words(trainw))

        return predict


    def sample(self, a, temperature=1.0):
        a = np.log(a + 1e-5) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))





