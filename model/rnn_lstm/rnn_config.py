# coding:utf-8

import numpy as np


class RNNConfig(object):

    def __init__(self):
        self.init_scale = 0.1
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.num_layers = 2
        self.num_steps = 20
        self.hidden_size = 200
        self.max_epoch = 4
        self.max_max_epoch = 13
        self.keep_prob = 1.0
        self.lr_decay = 0.5
        self.batch_size = 20
        self.vocabulary_size = 10000
        self.output_size = self.vocabulary_size
        self.embedding_init = np.array(np.random.normal(size=(self.vocabulary_size, self.hidden_size)), dtype=np.float32)

    def set_embedding_random(self, vocabulary_size, embedding_size):
        self.embedding_init = np.array(np.random.normal(size=(vocabulary_size, embedding_size)), dtype=np.float32)

    def set_embedding(self, embedding):
        self.embedding_init = np.array(embedding, dtype=np.float32)

class SmallConfig(RNNConfig):

    def __init__(self):
        """Small config."""
        RNNConfig.__init__(self)


class MediumConfig(RNNConfig):

    def __init__(self):
        """Medium config."""
        RNNConfig.__init__(self)
        self.learning_rate = 1.0
        self.init_scale = 0.05
        self.num_steps = 40
        self.hidden_size = 650
        self.max_epoch = 6
        self.max_max_epoch = 39
        self.keep_prob = 0.5
        self.lr_decay = 0.8


class LargeConfig(RNNConfig):
    def __init__(self):
        """Large config."""
        RNNConfig.__init__(self)
        self.init_scale = 0.04
        self.max_grad_norm = 10
        self.num_steps = 40
        self.hidden_size = 1500
        self.max_epoch = 14
        self.max_max_epoch = 55
        self.keep_prob = 0.35
        self.lr_decay = 1 / 1.15


class TinyConfig(RNNConfig):

    def __init__(self):
        """Tiny config for test"""
        RNNConfig.__init__(self)
        self.max_grad_norm = 1
        self.num_layers = 1
        self.num_steps = 2
        self.hidden_size = 2
        self.max_epoch = 1
        self.max_max_epoch = 1
        self.keep_prob = 1.0
        self.lr_decay = 0.5



