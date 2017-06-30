# coding:utf-8

import jieba as jb
import jieba.posseg as jp
import file_operation as fo
import sys
import tools


def _cut(line):
    words = []
    try:
        words = list(jp.cut(line))
    except Exception as e:
        print('cut line %s encounter an error %s' % (line, e))
    return words


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
        fo.operation_file(path, lambda x: _load_stop_word(x, sw=self.swlist), lambda x: x.endswith('.dic'))
        print('loaded stop word')

    def add_stop_word(self, word):
        if word.strip != '':
            self.swlist.add(word)

    def add_alternate_word(self, o_word, n_word):
        if o_word.strip != '' and n_word.strip != '':
            self.alternate_word[o_word] = n_word

    def load_user_dict(self, path):
        fo.operation_file(path, lambda x: jb.load_userdict(x), lambda x: x.endswith('.dic'))

    def load_alternate_word(self, path):

        def _load_alte_word(p, **kwds):
            alword = kwds['al']
            print('alte')
            for word in open(p, encoding='utf-8', mode='r').readlines():
                al = word.split(':')
                if len(al) == 2:
                    alword[al[0]] = al[1].strip()

        print('loading alternate word')
        fo.operation_file(path, lambda x: _load_alte_word(x, al=self.alternate_word), lambda x: x.endswith('.dic'))
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


def initial_dict(dict_path, stop_word_path):
    wl = WordLibrary()
    sys.setrecursionlimit(1000000)
    wl.load_user_dict(dict_path)
    wl.load_stop_word(stop_word_path)
    return wl

if __name__ == '__main__':
    import static_params as sp
    d = WordLibrary()
    d.load_alternate_word(sp.alternate_word_path)
    d.load_stop_word(sp.stop_word_path)
    print(d.alternate_word)
    print(d.swlist)
