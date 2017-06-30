import os
import pandas as pd
import file_operation as fo
from functools import reduce

class DataLoad:

    sep='\t'

    # def __init__(self):

    def loadfile_with_pandas_new(self, path, **kwds):
        names = kwds['names']
        f = path.split('\\')
        print('loading file path: %s' % path)
        data = pd.read_table(path, names=names, delimiter=self.sep, encoding='UTF-8')
        (m, _) = data.shape
        print('add %s, len %d' % (f[-1], m))
        kwds['data'].append(data)

    def load_by_pandas(self, path, names, sep):
        self.sep = sep
        data = []
        fo.operation_file(path, lambda x: self.loadfile_with_pandas_new(x, names=names, data=data))
        if len(data) < 2:
            result = data[0]
            (m, _) = result.shape
            print('total len %d' % m)
            return result
        else:
            result = reduce(lambda x, y: x.append(y), data)
            (m, _) = result.shape
            print('total len %d' % m)
        return result

    def load_text(self, path, sep=None):
        self.sep = sep
        data = []
        
        def text_file_load(path, **kwds):
            [kwds['data'].append(self._pre_line(line)) for line in open(path, encoding='utf-8', mode='r').readlines()]
            print('total len %d' % len(kwds['data']))
        fo.operation_file(path, lambda x: text_file_load(x, data=data))
        return data

    def _pre_line(self, line):
        if self.sep is not None:
            return line.strip().split(self.sep)
        else:
            return line.strip()


if __name__ == '__main__':
    d = DataLoad()
    n = ['tid', 'oid', 'num_iid', 'dp_id', 'valid_score', 'role',
             'nick', 'result', 'created', 'rated_nick', 'item_title',
             'item_price', 'content', 'reply']
