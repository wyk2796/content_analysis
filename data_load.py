import os
import pandas as pd


class DataLoad:

    sep='\t'

    # def __init__(self):

    def loadfile(self, path, names):
        return pd.read_table(path, names=names, delimiter=self.sep, encoding='UTF-8')

    def load(self, path, names, sep):
        self.sep = sep
        data = None
        if os.path.isdir(path):
            files = os.listdir(path)
            for f in files:
                load_path = path + '\\' + f
                try:
                    print('loading file path: %s' % load_path)
                    if data is None:
                        data = self.loadfile(load_path, names)
                        (m, _) = data.shape
                        print('first add %s, len %d' % (f, m))
                    else:
                        data = data.append(self.loadfile(load_path, names))
                        (m, _) = data.shape
                        print('add %s, len %d' % (f, m))
                except Exception as e:
                    print('loading file %s encounter an error:%s' % (load_path, e))
        else:
            data = self.loadfile(path, names)
        return data

if __name__ == '__main__':
    d = DataLoad()
    n = ['tid', 'oid', 'num_iid', 'dp_id', 'valid_score', 'role',
             'nick', 'result', 'created', 'rated_nick', 'item_title',
             'item_price', 'content', 'reply']
    fileData = d.load('E:\\temp\\transform_data', n, '#')
    print(fileData.shape)
    print(fileData)