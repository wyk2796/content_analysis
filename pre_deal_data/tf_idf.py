# coding:utf-8

import math
import tools


class TfIdf(object):

    def __init__(self):
        self.idf_list = {}
        self.total_word_count = 0
        self.total_content_count = 0

    def load_idf_from_file(self, path):
        with open(path, encoding='utf-8', mode='r') as data:
            for line in data.readlines():
                try:
                    couple = line.strip().split('\t')
                    if len(couple) == 2:
                        self.idf_list[couple[0]] = float(couple[1])
                    else:
                        self.idf_list[couple[0]] = 0
                except Exception as e:
                    print('read file: %s encounter an error %s' % (path, e))

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

    def computer_tf_idf(self, line, low_holder=0, up_holder=100):
        word_map = {}
        result = {}
        if isinstance(line, list):
            count = len(line)
            for w in line:
                word_map[w] = word_map.get(w, 0) + 1
            for (k, f) in word_map.items():
                grade = self._computer(k, f, count)
                if low_holder < grade < up_holder:
                    result[k] = grade
        return result

    def _computer(self, word, fp, count):
        return (fp / count) * self.idf_list.get(word, 0)

    def computer_idf(self, content):
        for line in content:
            for word in set(line):
                self.idf_list[word] = self.idf_list.get(word, 0) + 1
            self.total_content_count += 1

        for (k, v) in self.idf_list.items():
            self.idf_list[k] = math.log(self.total_content_count / v)

















