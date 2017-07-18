# coding:utf-8

import tensorflow as tf
import time
import numpy as np


class RNNSeq2Seq(object):

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
        self.states_seq2seq = None
        self.first_run = True
        self.forward_computer()
        if operation == 'Train':
            self.computer_cost()
            self.computer_train()
            self.computer_argmax_predict()
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
        # output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, self.config.hidden_size])

        outdata = self.decode(outputs)
        self.logits = tf.reshape(tf.stack(axis=1, values=outdata), [-1, self.config.output_size])
        # print(output.shape)

        # softmax_w = tf.get_variable("softmax_w", [self.config.hidden_size, self.config.output_size], dtype=tf.float32)
        # softmax_b = tf.get_variable("softmax_b", [self.config.output_size], dtype=tf.float32)
        # self.logits = tf.matmul(outdata, softmax_w) + softmax_b
        self._final_state = state
        self.first_run = False

    def decode(self, output):
        with tf.variable_scope('decode_seq2seq'):
            cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.config.num_layers)],
                                               state_is_tuple=True)
            if self.first_run:
                self.states_seq2seq = cell.zero_state(self.config.batch_size, dtype=tf.float32)
            attention_states = tf.get_variable("attention_states", [1,
                                                                    self.config.num_steps,
                                                                    self.config.output_size])
        (output_seq2seq, states_seq2seq) = tf.contrib.legacy_seq2seq.attention_decoder(output,
                                                                                       self.states_seq2seq,
                                                                                       attention_states,
                                                                                       cell,
                                                                                       output_size=self.config.output_size)
        self.states_seq2seq = states_seq2seq
        return output_seq2seq

    def layer(self, output, i):
        name = 'layer_%d' % i
        with tf.variable_scope(name):
            layer_w = tf.get_variable("layer_w", [self.config.hidden_size, self.config.hidden_size],
                                      dtype=tf.float32)
            layer_b = tf.get_variable("layer_b", [self.config.hidden_size], dtype=tf.float32)
        return tf.nn.relu(tf.matmul(output, layer_w) + layer_b)

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
        self.predict = tf.argmax(tf.nn.softmax(logits=self.logits), axis=1, name='predict_out')

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def load_embedding_from_word2vec(self, session, word2vec, path):
        try:
            word2vec.load_model(session, path)
            self.embeddings = word2vec.get_embedding()
        except Exception as e:
            print('load embedding encounter an error!', e)

    def train(self, session, train_data):
        start_time = time.time()
        iters = 0
        costs = 0
        epoch_size = train_data.size() // self.config.batch_size
        state = session.run(self._initial_state)
        fetch = {
            'train_op': self._train_op,
            "cost": self._cost,
            "final_state": self._final_state,
            "predict": self.predict,
        }
        count = [0, 0]
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

            y_p = vals["predict"]
            y_pp = np.reshape(y_p, newshape=(self.config.batch_size, self.config.num_steps))
            for g in range(self.config.batch_size):
                y_predict = [w for w in y_pp[g] if w != 0]
                labelw = [w for w in train_label[g] if w != 0]
                if list(y_predict) == list(labelw):
                    count[0] += 1
                else:
                    count[1] += 1
        print(count[0] + count[1], count[0] / (count[0] + count[1]), count[1] / (count[0] + count[1]))
        return np.exp(costs / iters)

    def run_predict(self, session, predict_data, wc):
        state = session.run(self._initial_state)
        fetches = {
            "predict": self.predict,
            "final_state": self._final_state,
        }
        predict = []
        count = [0, 0]
        out = open('E:\\temp\\model\\output.txt', encoding='utf-8', mode='w')
        for step, (train_input, label) in enumerate(predict_data.train_data_content(self.config.batch_size,
                                                                                    self.config.num_steps)):
            feed = {self.input_placeholder: train_input,
                    self._initial_state: state}
            vals = session.run(
                fetches, feed_dict=feed)
            y_p = vals["predict"]
            y_predict = [w for w in y_p if w != 0]
            labelw = [w for w in label[0] if w != 0]
            trainw = [w for w in train_input[0] if w != 0]
            state = vals["final_state"]
            if list(y_predict) == list(labelw):
                count[0] += 1
            else:
                count[1] += 1
            predict.append(y_predict)
            out.write('%s, %s, %s\n' % (str(wc.label_ids_to_words(y_predict)), str(wc.label_ids_to_words(labelw)), str(wc.ids_to_words(trainw))))
        out.close()
        print(count[0] + count[1], count[0] / (count[0] + count[1]), count[1] / (count[0] + count[1]))
        return predict

    def run_predict_real(self, session, predict_data, wc, outpath):
        state = session.run(self._initial_state)
        fetches = {
            "predict": self.predict,
            "final_state": self._final_state,
        }
        o = open(outpath, encoding='utf-8', mode='w')
        train_input, text = predict_data.predict_data_generate(self.config.num_steps)
        while train_input is not None:
            feed = {self.input_placeholder: [train_input],
                    self._initial_state: state}
            vals = session.run(
                fetches, feed_dict=feed)
            y_p = vals["predict"]
            y_predict = [w for w in list(y_p) if w != 0]
            state = vals["final_state"]
            predict_words = ''.join(wc.label_ids_to_words(y_predict))
            o.write('[%s]\t%s\n' % (predict_words.replace('END', ','), text))
            train_input, text = predict_data.predict_data_generate(self.config.num_steps)
        o.close()

    def sample(self, a, temperature=1.0):
        a = np.log(a + 1e-5) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))












