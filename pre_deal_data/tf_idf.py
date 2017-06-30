# coding:utf-8

import math
import tools
import file_operation as fo


class TfIdf(object):

    def __init__(self):
        self.idf_list = {}
        self.total_word_count = 0
        self.total_content_count = 0

    def idf_write_to_file(self, path):
        tools.dict_write_to_file(path, self.idf_list)

    def computer_tf_idf_with_org(self, content):

        """
        计算tf-idf
        :param content: 文本数据，分词后词数据：[['a','b','c'],['a','b'],['g'],[''r,'t','r']]
        :return:
        """
        result = []
        for line in content:
            result.append(self.computer_tf_idf(line))

    def computer_tf_idf(self, line, low_holder=0):
        word_map = {}
        result = {}
        if isinstance(line, list):
            count = len(line)
            for w in line:
                word_map[w] = word_map.get(w, 0) + 1
            for (k, f) in word_map.items():
                grade = self._computer(k, f, count)
                if grade > low_holder:
                    result[k] = grade
        return result

    def _computer(self, word, fp, count):
        return (fp / count) * self.idf_list.get(word, 0)

    def computer_idf(self):
        for (k, v) in self.idf_list.items():
            self.idf_list[k] = math.log(self.total_content_count / v)

    def computer_words_frp(self, content):
        for line in content:
            for word in set(line):
                self.total_word_count += 1
                self.idf_list[word] = self.idf_list.get(word, 0) + 1
            self.total_content_count += 1

    def load_local_words_frp(self, path, content_num):
        self.idf_list = fo.load_tuple_text_from_file(path)
        self.total_content_count = content_num
        self.total_word_count = len(self.idf_list)
        print('words num : %f' % self.total_word_count)
        return self

    def load_local_idf(self, path):
        self.idf_list = fo.load_tuple_text_from_file(path)















