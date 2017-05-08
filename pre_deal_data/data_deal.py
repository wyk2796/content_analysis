# coding:utf-8
import wordseg as ws


def framer_to_apriori(data, swlist=None):
    def createrdata(line):
        ll = []
        [ll.append(w) for w in ws.word_cut(line, swlist)]
        return ll
    data_list = []
    data.apply(func=lambda x: data_list.append(createrdata(str(x['content']))), axis=1)
    return data_list


def list_from_frame(frame_data, column_name):
    return list(frame_data[column_name])


def change_sign_in_content(line):
    sign_list = ['，', '。', '！', '？', ' ', '\t', '.', '…']
    for sign in sign_list:
        line = line.replace(sign, ',')
    ','.join(filter(lambda l: l != '', line.split(',')))
    return line


def list_from_frame_content_train(frame_data, column_name):
    data_ll = []
    frame_data.apply(func=lambda x: [data_ll.append(line.strip())
                                     for line in change_sign_in_content(str(x[column_name])).split(',')], axis=1)
    return data_ll


def list_from_frame_content(frame_data, column_name):
    data_ll = []
    frame_data.apply(func=lambda x: data_ll.append(str(x[column_name]).strip()), axis=1)
    return data_ll


def statistic_word_count(data, wl=None):

    def word_count(word, mapw):
        mapw[word] = mapw.get(word, 0) + 1

    dictory = {}
    data.apply(func=lambda x: [word_count(w, dictory) for w in wl.word_cut_with_sign(str(x['content']))], axis=1)
    return dictory

