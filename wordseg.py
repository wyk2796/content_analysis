# coding:utf-8

import jieba as jb
import jieba.posseg as jp
import os
import tools


def operation_single_file(path, func, *args, **kwds):
    try:
        print('operation file path: %s' % path)
        func(path, *args, **kwds)
    except Exception as e:
        print('operation file %s encounter an error:%s' % (path, e))


def operation_file(path, func, *args, **kwds):
    if os.path.isdir(path):
        [operation_single_file(path + '\\' + f_name, func, *args, **kwds)
         for f_name in os.listdir(path) if f_name.endswith('.dic')]
    else:
        operation_single_file(path, func, *args, **kwds)


def load_user_dict(path):

    """
    加载path下面的字典文件
    :param path: 字典文件夹
    :return:
    """
    print('loading dict word')
    operation_file(path, lambda x: jb.load_userdict(x))
    print('loaded dict word')


def load_stop_word(path):

    """
    加载停留词，
    :param path:停留词文件路径
    :param kwds: 存储停留词参数
    :return:
    """

    def _load_stop_word(p, **kwds):
        [kwds['sw'].add(word.strip()) for word in open(p, encoding='utf-8', mode='r').readlines()]

    words = set()
    words.add(' ')
    print('loading stop word')
    operation_file(path, lambda x: _load_stop_word(x, sw=words))
    print('loaded stop word')
    return words


def _cut(line):
    words = []
    try:
        words = list(jp.cut(line))
    except Exception as e:
        print('cut line %s encounter an error %s' % (line, e))
    return words


# def sign_word(line, swlist=None):
#     words = {}
#
#     def add(w, sign_map):
#         sign_map[w.word] = w.falg
#     if swlist is not None and len(swlist) > 0:
#         [add(word, words) for word in jb.cut(line) if word not in swlist]
#     else:
#         [add(word, words) for word in jb.cut(line)]
#     return words


def computer_word_fp(content):
    words_map = {}
    for line in content:
        for word in line:
            words_map[word] = words_map.get(word, 0) + 1
    return words_map


def create_dict(path, content):
    tools.dict_write_to_file(path, computer_word_fp(content))


class WordLibrary(object):

    def __init__(self):
        self.swlist = set()
        self.alternate_word = {}

    def is_empty_swlist(self):
        return len(self.swlist) == 0

    def is_empty_alter(self):
        return len(self.alternate_word) == 0

    def load_stop_word(self, path):
        """
        加载停留词，
        :param path:停留词文件路径
        :param kwds: 存储停留词参数
        :return:
        """

        def _load_stop_word(p, **kwds):
            [kwds['sw'].add(word.strip()) for word in open(p, encoding='utf-8', mode='r').readlines()]

        self.swlist.add(' ')
        print('loading stop word')
        operation_file(path, lambda x: _load_stop_word(x, sw=self.swlist))
        print('loaded stop word')

    def add_stop_word(self, word):
        if word.strip != '':
            self.swlist.add(word)

    def add_alternate_word(self, o_word, n_word):
        if o_word.strip != '' and n_word.strip != '':
            self.alternate_word[o_word] = n_word

    def load_alternate_word(self, path):

        def _load_alte_word(p, **kwds):
            alword = kwds['al']
            print('alte')
            for word in open(p, encoding='utf-8', mode='r').readlines():
                al = word.split(':')
                if len(al) == 2:
                    alword[al[0]] = al[1].strip()

        print('loading alternate word')
        operation_file(path, lambda x: _load_alte_word(x, al=self.alternate_word))
        print('loading alternate word')

    def is_in_stop_list(self, word):
        return word in self.swlist

    def alternate_word(self, word):
        return self.alternate_word.get(word, word)

    def word_cut_with_sign(self, line):
        words = []
        if not self.is_empty_swlist():
            [words.append(w.word) for w in _cut(line) if
             isinstance(w.word, str) and w.word not in self.swlist]
        else:
            [words.append(w.word) for w in _cut(line) if isinstance(w.word, str)]
        return words




if __name__ == '__main__':
    import static_params as sp
    d = WordLibrary()
    d.load_alternate_word(sp.alternate_word_path)
    d.load_stop_word(sp.stop_word_path)
    print(d.alternate_word)
    print(d.swlist)
