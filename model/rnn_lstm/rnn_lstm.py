# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

import numpy as np
import tensorflow as tf
from model.rnn_lstm import pti
import model.rnn_lstm.rnn_config as conf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string("data_path", 'E:\\temp\data\simple-examples\simple-examples\data',
                    "Where the training/test data is stored.")

flags.DEFINE_string("save_path", 'E:\\temp\data\model_test',
                    "Model output directory.")

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
  """The PTB model."""

  def __init__(self, operation, config):
    self.config = config
    self.add_placeholder()
    self.operation = operation
    self.batch_size = self.config.batch_size
    self.num_steps = self.config.num_steps
    self.hidden_size = self.config.hidden_size
    self.vocab_size = self.config.vocab_size
    self.forword_computer()
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    if operation == 'Train':
        self.computer_cost()
        self.computer_train()
    if operation == 'Valid':
        self.computer_cost()
    if operation == 'Predict':
        self.computer_predict()

  def lstm_cell(self):
      return tf.contrib.rnn.BasicLSTMCell(
              self.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

  def forword_computer(self):
      attn_cell = self.lstm_cell
      if self.operation == 'Train' and self.config.keep_prob < 1:
          def attn_cell():
              return tf.contrib.rnn.DropoutWrapper(
                  self.lstm_cell(), output_keep_prob=self.config.keep_prob)
      cell = tf.contrib.rnn.MultiRNNCell(
          [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
      self._initial_state = cell.zero_state(self.batch_size, data_type())
      with tf.device("/cpu:0"):
          embedding = tf.get_variable(
              "embedding", [self.vocab_size, self.hidden_size], dtype=data_type())
          inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
      if self.operation == 'Train' and self.config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, self.config.keep_prob)
      self.outputs = []
      state = self._initial_state
      with tf.variable_scope("RNN"):
          for time_step in range(self.num_steps):
              if time_step > 0:
                  tf.get_variable_scope().reuse_variables()
              (cell_output, state) = cell(inputs[:, time_step, :], state)
              self.outputs.append(cell_output)
      output = tf.reshape(tf.stack(axis=1, values=self.outputs), [-1, self.hidden_size])
      softmax_w = tf.get_variable(
          "softmax_w", [self.hidden_size, self.vocab_size], dtype=data_type())
      softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=data_type())
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

      self._new_lr = tf.placeholder(
          tf.float32, shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)

  def computer_cost(self):
      loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
          [self.logits],
          [tf.reshape(self.labels_placeholder, [-1])],
          [tf.ones([self.batch_size * self.num_steps], dtype=data_type())])
      self._cost = tf.reduce_sum(loss) / self.batch_size

  def computer_predict(self):
      self.predict = tf.nn.softmax(logits=self.logits)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def add_placeholder(self):
      self.input_placeholder = tf.placeholder(
      tf.int32, shape=[None, self.config.num_steps], name='Input')
      self.labels_placeholder = tf.placeholder(
      tf.int32, shape=[None, self.config.num_steps], name='Target')

  def run_epochs(self, session, data, train_op=None, verbose=False):
      config = self.config
      start_time = time.time()
      epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
      costs = 0.0
      iters = 0
      state = session.run(self.initial_state)
      fetch = {
        "cost": self.cost,
        "final_state": self.final_state
      }
      if train_op is not None:
        fetch['train_op'] = train_op
      for step, (x, y) in enumerate(
              pti.ptb_iterator(data, config.batch_size, config.num_steps)):
        # We need to pass in the initial state and retrieve the final state to give
        # the RNN proper history
        feed = {self.input_placeholder: x,
                self.labels_placeholder: y,
                self.initial_state: state}
        vals = session.run(
          fetch, feed_dict=feed)  # 用RNN的final state，计算loss，并用train训练到最小loss
        costs += vals['cost']
        state = vals['final_state']
        iters += self.num_steps
        if verbose and step % (epoch_size // 10) == 10:
          print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 iters * config.batch_size / (time.time() - start_time)))
      return np.exp(costs / iters)

  def run_predict(self, session, data, stop_num=100):
    state = session.run(self.initial_state)
    predict_data = data
    print(data)
    for j in range(stop_num):
        fetches = {
          "predict": self.get_predict,
          "final_state": self.final_state,
        }
        feed_dict = {
          self.input_placeholder: [predict_data[-1:]],
          self.initial_state: state
        }
        vals = session.run(fetches, feed_dict)
        y_predict = vals["predict"][-1]
        print(y_predict)
        new_ids = pti.sample(y_predict, 1.0)
        predict_data.append(new_ids)
        state = vals["final_state"]
        self.batch_size = 1
        self.num_steps = len(predict_data)
        print(predict_data)

    return predict_data


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def get_predict(self):
    return self.predict


def main():
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  word_IDs = pti.WordsIDs(FLAGS.data_path + '\ptb.train.txt')
  train_data, valid_data, test_data, _ = word_IDs.ptb_raw_data(FLAGS.data_path)

  predict_words = ['i', 'love', 'you']
  predict_ids = word_IDs.words_to_ids(predict_words)

  config = conf.get_config(FLAGS)
  config.max_max_epoch = 1
  eval_config = conf.get_config(FLAGS)
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  eval_config.keep_prob = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel('Train', config=config)
      # tf.summary.scalar("Training Loss", m.cost)
      # tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel('Valid', config=config)
      # tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Predict"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mpredict = PTBModel('Predict', config=eval_config)


    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = m.run_epochs(session, train_data, train_op=m.train_op, verbose=True)

        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = mvalid.run_epochs(session, valid_data)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      words = mpredict.run_predict(session, predict_ids, 100)
      print(word_IDs.ids_to_word(words))

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    main()
