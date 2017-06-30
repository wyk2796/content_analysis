# coding:utf-8

import tensorflow as tf
import static_params as params
import wordseg as ws
import sys
import numpy as np
import model.train_data_generate as wg
import model.words_deal as wd
import data_load as dl
import math


class Word2vcConfig:

    def __init__(self, vocabulary_size=50000, batch_size=128, embedding_size=128,
                 skip_window=1, num_skips=2, num_sampled=64):
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self. num_skips = num_skips
        self.num_sampled = num_sampled
        self.count = 0


class Word2Vec:

    def __init__(self, config):
        self.conf = config
        self.train_labels = None
        self.train_inputs = None
        self.loss = None
        self.optimizer = None
        self.graph = tf.Graph()
        self.create_graph()

    def create_graph(self):
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.conf.batch_size], name='input/data')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.conf.batch_size, 1], name='input/label')
            with tf.device('/cpu:0'):
                with tf.variable_scope('word2vec', reuse=None):
                    embeddings = tf.get_variable("embedding",
                                                 initializer=tf.random_uniform([self.conf.vocabulary_size,
                                                                                self.conf.embedding_size], -1.0, 1.0),
                                                 dtype=tf.float32)
                    embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                    nce_weights = tf.get_variable('nce_weight',
                                                  initializer=tf.truncated_normal([self.conf.vocabulary_size,
                                                                                   self.conf.embedding_size],
                                                                                  stddev=1.0 / math.sqrt(
                                                                                      self.conf.embedding_size)))
                    nce_biases = tf.get_variable('nce_biases', initializer=tf.zeros([self.conf.vocabulary_size]))
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, self.train_labels, embed,
                               self.conf.num_sampled, self.conf.vocabulary_size))
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

    def run_train(self, session, input_data, label_data):
        _, loss_data = session.run([self.optimizer, self.loss],
                                   {self.train_inputs: input_data, self.train_labels: label_data})
        return loss_data

    def run_valid(self, session, input_data, label_data):
        _, loss_data = session.run(self.loss, {self.train_inputs: input_data, self.train_labels: label_data})
        return loss_data

    def get_embedding(self):
        with tf.variable_scope('word2vec', reuse=True):
            embeddings = tf.get_variable("embedding",
                                         initializer=tf.random_uniform([self.conf.vocabulary_size,
                                                                        self.conf.embedding_size], -1.0, 1.0),
                                         dtype=tf.float32)
        return embeddings

    def similary(self, session, data, simi_num):

        embeddings = self.get_embedding()
        simi_data = tf.constant(data, dtype=tf.int32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, simi_data)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)
        sim = session.run(similarity)
        return [(-sim[i, :]).argsort()[1:simi_num+1] for i in range(len(data))]

    def load_model(self, session, path):
        sv = tf.train.Saver(allow_empty=True)
        try:
            path = tf.train.get_checkpoint_state(path)
            sv.restore(session, path.model_checkpoint_path)
        except Exception as e:
            print('load model encounter an error', e)

    def get_embedding_value(self, session):
        return session.run(self.get_embedding())

    def save_model(self, session, path):
        saver = tf.train.Saver(allow_empty=True)
        saver.save(session, path)


