# coding:utf-8
import numpy as np
import collections
import pre_deal_data.data_deal as pd

class W2CGenerateBatch(object):

    def __init__(self, data_content, batch_size, num_skips, skip_window):
        assert num_skips <= 2 * skip_window
        self.data_index = 0
        self.data_len = data_content.line_size()
        self.data = data_content.data
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.words = []
        self.labels = []

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def set_num_skips(self, num_skips):
        self.num_skips = num_skips
        return self

    def set_skip_window(self, skip_window):
        self.skip_window = skip_window
        return self

    def generate_batch(self):
        batch = np.ndarray(shape=self.batch_size, dtype=np.int32)  # batch_size: the number of words in one batch
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        if len(self.words) > self.batch_size:
            for i in range(self.batch_size):
                batch[i] = self.words[0]
                labels[i, 0] = self.labels[0]
                del self.words[0]
                del self.labels[0]
        else:
            while len(self.words) < self.batch_size:
                self.line_couple()
            for i in range(self.batch_size):
                batch[i] = self.words[0]
                labels[i, 0] = self.labels[0]
                del self.words[0]
                del self.labels[0]
        return batch, labels  # batch: ids [batch_size] lebels:ids [batch_size*1]

    def line_couple(self):
        word = self.data[self.data_index % self.data_len]
        words_num = len(word)
        if words_num > 2:
            for index in range(len(word)):
                for i in range(index - self.skip_window, index + self.skip_window + 1):
                    if 0 <= i < words_num and i is not index and word[index] is not 0 and word[i] is not 0:
                        self.words.append(word[index])
                        self.labels.append(word[i])
        self.data_index += 1


class RNNGenerateBatch(object):

    def __init__(self, data_content, config):
        self.data = data_content
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.size = len(data_content.data)

    def generate_batch(self):
        raw_data = np.array(self.data.data, dtype=np.int32)
        data_len = len(self.data.data)
        batch_len = data_len // self.batch_size
        data = np.zeros([self.batch_size, batch_len], dtype=np.int32)
        for i in range(self.batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
        epoch_size = (batch_len - 1) // self.num_steps
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
        for i in range(epoch_size):
            x = data[:, i * self.num_steps:(i + 1) * self.num_steps]
            y = data[:, i * self.num_steps + 1:(i + 1) * self.num_steps + 1]
            yield (x, y)

    def sample(self, a, temperature=1.0):
        a = np.log(a + 1e-5) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))








