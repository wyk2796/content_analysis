# coding:utf-8
import numpy as np
from model import train_data_generate as tdg
import static_params as params
import tensorflow as tf
from model.word2vec import word2vcmodel
from data_load import DataLoad
from model import words_deal as wd

def computer_word2vce(wc, re_Train = False):
    batch_size = 128
    embedding_size = params.embedding_size
    skip_window = 2
    num_skips = 4
    num_sampled = 64

    train_data = tdg.W2CGenerateBatch(wc, batch_size, num_skips, skip_window)

    conf_word2vec = word2vcmodel.Word2vcConfig(wc.vocabulary_size,
                                               batch_size,
                                               embedding_size,
                                               skip_window,
                                               num_skips,
                                               num_sampled)
    word2vec = word2vcmodel.Word2Vec(conf_word2vec)

    average_loss = 0
    with tf.Session(graph=word2vec.graph) as session:
        if re_Train:
            word2vec.load_model(session, params.words2vec_model_path)
        else:
            tf.global_variables_initializer().run()

        for i in range(100000):
            batch, lable = train_data.generate_batch()
            average_loss += word2vec.run_train(session, batch, lable)
            if i % 1000 == 0:
                print('%d loss %.5f' % (i, average_loss // 1000))
                average_loss = 0

            if i % 2000 == 0:
                word2vec.save_model(session, params.xtep_words2vec_model)

        top_k = 10
        valid_examples = np.array([11, 12, 13, 14,
                                   15, 16, 17, 18,
                                   19, 20, 21, 22,
                                   24, 25, 26, 27])
        print(valid_examples)
        sim = word2vec.similary(session, valid_examples, top_k)
        for j in range(16):
            word = wc.ids_words[valid_examples[j]]
            nest = sim[j]
            log_str = "Nearest to %s:" % word
            for k in range(top_k):
                close_word = wc.ids_words[nest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)


def create_word2vec():
    d = DataLoad()
    good_data = d.load_text(params.xtep_good_seg_words_path, ',')
    bad_data = d.load_text(params.xtep_bad_seg_words_path, ',')
    good_data_anta = d.load_text(params.anta_good_seg_words_path, ',')
    bad_data_anta = d.load_text(params.anta_bad_seg_words_path, ',')
    good_data.extend(bad_data)
    good_data.extend(good_data_anta)
    good_data.extend(bad_data_anta)
    data = sorted(good_data, key=lambda x: len(x))
    wc = wd.create_content_data(data, params.vocabulary_size, params.xtep_words_ids)
    computer_word2vce(wc, re_Train=False)


if __name__ == '__main__':
    create_word2vec()

