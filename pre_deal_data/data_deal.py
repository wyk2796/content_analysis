# coding:utf-8

import static_params as params
from data_load import DataLoad
from pre_deal_data import separate_data as sd
import wordseg as ws


def list_from_frame(frame_data, column_name):
    return list(frame_data[column_name])


def change_sign_in_content(line):
    sign_list = ['，', '。', '！', '？', ' ', '\t', '.', '…', '：', '!']
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
    frame_data.apply(func=lambda x: data_ll.append(change_sign_in_content(str(x[column_name])).strip()), axis=1)
    return data_ll


def statistic_word_count(data, wl=None):

    def word_count(word, mapw):
        mapw[word] = mapw.get(word, 0) + 1

    dictory = {}
    data.apply(func=lambda x: [word_count(w, dictory) for w in wl.word_cut_with_sign(str(x['content']))], axis=1)
    return dictory


def statistic_word_count_text(data, wl=None):
    dictory = {}

    def word_count(word, mapw):
        mapw[word] = mapw.get(word, 0) + 1

    [word_count(w, dictory) for x in data for w in wl.word_cut_with_sign(x)]
    return dictory


def statistic_word_from_line(data):

    dictory = {}

    def word_count(word, mapw):
        mapw[word] = mapw.get(word, 0) + 1
    [word_count(w, dictory) for x in data for w in x]
    return dictory


def pre_seg_words(words):
    word_output = []
    for w in words:
        if w.isalnum() and not w.isdigit() and not w.isspace():
            word_output.append(w)
    return word_output


def pre_xtep():
    d = DataLoad()
    good_data = d.load_text(params.xtep_good_content_path)
    bad_data = d.load_text(params.xtep_bad_middle_content_path)
    wl = ws.initial_dict(params.dict_path, params.stop_word_path)

    with open(params.xtep_bad_seg_words_path, mode='w', encoding='utf-8') as out:
        for line in bad_data:
            words = pre_seg_words(wl.word_cut_with_sign(line))
            if len(words) >= 4:
                line_w = ','.join(words)
                out.write(line_w + '\n')

    with open(params.xtep_good_seg_words_path, mode='w', encoding='utf-8') as out:
        for line in good_data:
            words = pre_seg_words(wl.word_cut_with_sign(line))
            if len(words) >= 4:
                line_w = ','.join(words)
                out.write(line_w + '\n')


def pre_anta_data():
    d = DataLoad()
    original_data = d.load_text(params.anta, ',')
    sd.separate_data_to(original_data, lambda x: x[7],
                        params.anta_good_path,
                        params.anta_good_content_path,
                        params.anta_bad_middel_path,
                        params.anta_bad_middle_content_path)
    pre_anta_segword()


def pre_anta_segword():
    d = DataLoad()
    good_data = d.load_text(params.anta_good_content_path)
    bad_data = d.load_text(params.anta_bad_middle_content_path)
    wl = ws.initial_dict(params.dict_path, params.stop_word_path)
    good_data_sorted = sorted(good_data, key=lambda x: len(x))
    bad_data_sorted = sorted(bad_data, key=lambda x: len(x))

    only_good_data = set(good_data_sorted)
    only_bad_data = set(bad_data_sorted)

    with open(params.anta_bad_seg_words_path, mode='w', encoding='utf-8') as out:
        for line in only_bad_data:
            words = pre_seg_words(wl.word_cut_with_sign(line))
            if len(words) >= 4:
                line_w = ','.join(words)
                out.write(line_w + '\n')

    with open(params.anta_good_seg_words_path, mode='w', encoding='utf-8') as out:
        for line in only_good_data:
            words = pre_seg_words(wl.word_cut_with_sign(line))
            if len(words) >= 4:
                line_w = ','.join(words)
                out.write(line_w + '\n')


if __name__ == '__main__':
    pre_xtep()
    pre_anta_segword()