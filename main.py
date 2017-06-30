# coding:utf-8

from data_load import DataLoad
import static_params as params
import wordseg as ws
from pre_deal_data import data_deal
from pre_deal_data import tf_idf
from fpgrowth import computer_freqitem as cf
from fpgrowth import fp_growth_00 as fg0
import sys

def run():
    wl = ws.initial_dict(params.dict_path, params.stop_word_path)
    bad_data = initial_pandas(params.content_bad_path)
    content_data = data_deal.list_from_frame_content_train(bad_data, 'content')
    # print(len(content_data))
    con_words = [wl.word_cut_with_sign(w) for w in content_data]
    tfidf = initial_tf_idf(wl)
    fp_bad = computer_fp_data(con_words)
    write(params.fp_words, fp_bad)
    predict_content(bad_data, fp_bad, tfidf, wl)


def computer_fp_data(data):
    bad_fp = []
    for (itemset, support) in fg0.find_frequent_itemsets(data, 50, True):
        bad_fp.append(itemset)
    return bad_fp


def predict_content(data, fp_data, tfidf, wl):
    result = {}
    o1 = open('E:\\temp\\result_tmp.txt', encoding='utf-8', mode='w')
    for com in data_deal.list_from_frame_content(data, 'content'):
        com_fp = []
        # sw_dict = {}
        for sen in com.split(','):
            ww = wl.word_cut_with_sign(sen)
            idf_map = tfidf.computer_tf_idf(ww, 2)
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
                o1.write('content:%s, seg_word:%s, fp_word: %s, idf_set:%s\n' % (com, str(ww), str(com_fp), str(idf_map)))
        result[com] = com_fp
    write('E:\\temp\\result.txt', result)


def initial_tf_idf(wl, train=False):
    tfidf = tf_idf.TfIdf()
    tfidf.load_local_idf(params.tf_idf_model)
    idf_len = len(tfidf.idf_list)
    print(idf_len)
    if idf_len > 0 and train:
        # d = initial_text(params.content_orginal_path)
        # data = list(map(lambda x: x[12], d))
        # print('computer words frequent')
        # word_fp = data_deal.statistic_word_count_text(data, wl)
        # print('stastic finish')
        # ssw = sorted(word_fp.items(), key=lambda x: x[1], reverse=True)
        # print('order finish')
        # tools.tuple_list_write_to_file(params.dict_file, ssw)

        # con_words = [wl.word_cut_with_sign(w) for w in data]
        """create dict """
        print('computer idf')
        tfidf.load_local_words_frp(params.statistic_words, 5310917).computer_idf()
        tfidf.idf_write_to_file(params.tf_idf_model)
    return tfidf


def write(path, wmap):
    print('write to %s' % path)
    with open(path, encoding='utf-8', mode='w') as out:
        if isinstance(wmap, dict):
            [out.write(str(x) + ':' + str(k) + '\n') for (x, k) in wmap.items()]
        elif isinstance(wmap, list):
            [out.write(str(line) + '\n') for line in wmap]
    print('finish write %s' % path)


def initial_pandas(path):
    d = DataLoad()
    return d.load_by_pandas(path, params.table_name, '#')


def initial_text(path):
    d = DataLoad()
    return d.load_text(path, '#')


#if __name__ == '__main__':
    # create_rnn_train()
    # pre_anta_data()
    # per_step()
    #create_word2vec()
    # run()

