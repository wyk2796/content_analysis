# coding:utf-8

from data_load import DataLoad
import static_params as params
import wordseg as ws
from pre_deal_data import data_deal
from pre_deal_data import tf_idf
from fpgrowth import computer_freqitem as cf
from fpgrowth import fp_growth_00 as fg0
import sys
import tools


def run():
    wl = initial_dict()
    # bad_data = initial(params.content_text_path)
    # content_data = data_deal.list_from_frame_content_train(bad_data, 'content')
    # print(len(content_data))
    # con_words = [wl.word_cut_with_sign(w) for w in content_data]
    tfidf = initial_tf_idf(wl)


def computer_fp_data(data):
    bad_fp = []
    for (itemset, support) in fg0.find_frequent_itemsets(data, 100, True):
        bad_fp.append(itemset)


def predict_content(data, fp_data, tfidf, wl):
    result = {}
    o1 = open('E:\\temp\\result_tmp.txt', encoding='utf-8', mode='w')
    for com in data_deal.list_from_frame_content(data, 'content'):
        com_fp = []
        # sw_dict = {}
        for sen in com.split(','):
            ww = wl.word_cut_with_sign(sen)
            idf_map = tfidf.computer_tf_idf(ww, 2, 11)
            simple_word = set(ww)
            # for sw in simple_word:
            #     sw_dict[sw] = idf_map.get(sw, 0) + 0
            # ssw = sorted(sw_dict.items(), key=lambda x: x[1], reverse=True)
            lenw = len(simple_word)
            sen_fp = []
            for fp in fp_data:
                tmp = ww.copy()
                tmp.extend(fp)
                unit = set(tmp)
                if lenw == len(unit):
                    # com_fp.append(fp)
                    sen_fp.extend(fp)
            if len(sen_fp) != 0:
                sen_fp.extend(idf_map.keys())
                sen_fp = list(set(sen_fp))
                nsw = []
                for nw in ww:
                    if nw in sen_fp and nw not in nsw:
                        nsw.append(str(nw))

                if nsw != '':
                    com_fp.append(''.join(nsw))
                o1.write(com + ':' + str(ww)+':' + str(com_fp) + ':' + str(idf_map) + '\n')
        result[com] = com_fp
    write('E:\\temp\\result.txt', result)

def initial_tf_idf(wl):
    tfidf = tf_idf.TfIdf()
    # tfidf.load_idf_from_file(params.tf_idf_model)
    # idf_len = len(tfidf.idf_list)
    # print(idf_len)
    if True:
        d = initial(params.content_orginal_path)
        print('computer words frequent')
        tools.dict_write_to_file(params.dict_file, data_deal.statistic_word_count(d, wl))

        con_words = [wl.word_cut_with_sign(w) for w in data_deal.list_from_frame_content(d, 'content')]
        """create dict """
        print('computer idf')
        tfidf.computer_idf(con_words)
        tfidf.idf_write_to_file(params.tf_idf_model)
    return tfidf


def run_fp():

    swlist = initial_dict()

    good_data = initial(params.content_good_path)
    bad_data = initial(params.content_bad_path)
    middle_data = initial(params.content_middle_path)

    # good_count = data_deal.statistic_word_count(good_data, swlist)
    # bad_count = data_deal.statistic_word_count(bad_data, swlist)
    # middle_count = data_deal.statistic_word_count(middle_data, swlist)
    #
    # write(params.middle_table_path + '\\' + 'good_word_count.txt', good_count)
    # write(params.middle_table_path + '\\' + 'bad_word_count.txt', bad_count)
    # write(params.middle_table_path + '\\' + 'middle_word_count.txt', middle_count)

    good_fp = cf.computer_apri(data_deal.framer_to_apriori(good_data, swlist), 9000)
    bad_fp = cf.computer_apri(data_deal.framer_to_apriori(bad_data, swlist), 100)
    middle_fp = cf.computer_apri(data_deal.framer_to_apriori(middle_data, swlist), 100)

    write(params.middle_table_path + '\\' + 'good_fp.txt', good_fp)
    write(params.middle_table_path + '\\' + 'bad_fp.txt', bad_fp)
    write(params.middle_table_path + '\\' + 'middle_fp.txt', middle_fp)


def write(path, wmap):
    print('write to %s' % path)
    with open(path, encoding='utf-8', mode='w') as out:
        if isinstance(wmap, dict):
            [out.write(str(x) + ':' + str(k) + '\n') for (x, k) in wmap.items()]
        elif isinstance(wmap, list):
            [out.write(str(line) + '\n') for line in wmap]
    print('finish write %s' % path)


def initial_dict():
    wl = ws.WordLibrary()
    sys.setrecursionlimit(1000000)
    ws.load_user_dict(params.dict_path)
    wl.load_stop_word(params.stop_word_path)
    return wl


def initial(path):
    d = DataLoad()
    return d.load(path, params.table_name, '#')

if __name__ == '__main__':
    run()

