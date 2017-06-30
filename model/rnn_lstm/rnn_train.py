# coding:utf-8
import numpy as np
from model.rnn_lstm import rnn_config
from model import train_data_generate as tdg
import static_params as params
import tensorflow as tf
from model.rnn_lstm import rnn_model
from model.rnn_lstm import rnn_short_text_model as rstm
from model.word2vec import word2vcmodel
from model import words_deal
from data_load import DataLoad

def RNN_train(wc):

    conf = rnn_config.MediumConfig()
    conf.vocabulary_size = wc.vocabulary_size
    conf.max_max_epoch = 5
    conf.embedding_init = get_embedding_from_word2vec(params.xtep_words2vec_model)
    #conf.set_embedding_random(wc.vocabulary_size, conf.hidden_size)
    rnn_generate = tdg.RNNGenerateBatch(wc.data_reshape_to_one(), conf)

    predict_conf = rnn_config.MediumConfig()
    predict_conf.vocabulary_size = wc.vocabulary_size
    predict_conf.batch_size = 1
    predict_conf.num_steps = 1
    predict_conf.keep_prob = 1
    predict_conf.embedding_init = conf.embedding_init

    predict_word = ['鞋']
    predict_ids = wc.words_to_ids(predict_word)
    graph = tf.Graph()

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-conf.init_scale,
                                                    conf.init_scale)
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                train_model = rnn_model.RNNModel(conf, 'Train')

        with tf.name_scope('Predict'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                predict_model = rnn_model.RNNModel(predict_conf, 'Predict')

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        for i in range(conf.max_max_epoch):
            lr_decay = conf.lr_decay ** max(i + 1 - conf.max_epoch, 0.0)
            train_model.assign_lr(session, conf.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model._lr)))
            train_perplexity = train_model.train(session, rnn_generate)

            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            # valid_perplexity = mvalid.run_epochs(session, valid_data)
            # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            if params.xtep_rnn_model:
                print("Saving model to %s." % params.xtep_rnn_model)
                sv = tf.train.Saver()
                sv.save(session, params.xtep_rnn_model)

        words = predict_model.run_predict(session, predict_ids, rnn_generate)
        print(''.join(wc.ids_to_words(words)).replace('END', '\n'))


def get_embedding_from_word2vec(path):
    batch_size = 128
    embedding_size = params.embedding_size
    skip_window = 1
    num_skips = 2
    num_sampled = 64
    vocabulary_size = params.vocabulary_size
    conf_word2vec = word2vcmodel.Word2vcConfig(vocabulary_size,
                                               batch_size,
                                               embedding_size,
                                               skip_window,
                                               num_skips,
                                               num_sampled)
    word2vec = word2vcmodel.Word2Vec(conf_word2vec)
    with tf.Session(graph=word2vec.graph) as session:
        word2vec.load_model(session, path)
        print('load successful')
        embedding = word2vec.get_embedding_value(session)
    return embedding


def random_data(data, num):
    indexs = np.random.choice(1000000, num, replace=False)
    return [data[index] for index in indexs]


def create_rnn_train():
    d = DataLoad()
    good_data = d.load_text(params.xtep_good_seg_words_path, ',')
    print(len(good_data))
    bad_data = d.load_text(params.xtep_bad_seg_words_path, ',')
    good_sample = random_data(good_data, 100000)
    good_sample.extend(bad_data)
    data = sorted(good_sample, key=lambda x: len(x))
    print(len(data))
    wc = words_deal.WordsContent(params.vocabulary_size)
    wc.load_word_index(params.xtep_words_ids)
    wc.add_content(data)
    RNN_train(wc)



def RNN_train_2(td, wc, re_train = False):
    conf = rnn_config.MediumConfig()
    conf.vocabulary_size = wc.vocabulary_size
    conf.hidden_size = params.embedding_size
    conf.num_steps = 30
    conf.max_max_epoch = 100
    conf.max_epoch = 90
    conf.learning_rate = 0.005
    conf.lr_decay = 0.9
    conf.output_size = wc.label_size()
    conf.embedding_init = get_embedding_from_word2vec(params.xtep_words2vec_model)
    #conf.set_embedding_random(wc.vocabulary_size, conf.hidden_size)

    predict_conf = rnn_config.MediumConfig()
    predict_conf.vocabulary_size = wc.vocabulary_size
    predict_conf.hidden_size = params.embedding_size
    predict_conf.batch_size = 1
    predict_conf.num_steps = 30
    predict_conf.keep_prob = 1
    predict_conf.output_size = wc.label_size()
    print('output size', wc.label_size())
    predict_conf.embedding_init = conf.embedding_init

    graph = tf.Graph()

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-conf.init_scale,
                                                    conf.init_scale)
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                train_model = rstm.RNNModel(conf, 'Train')

        with tf.name_scope('Predict'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                predict_model = rstm.RNNModel(predict_conf, 'Predict_argmax')

    sv = tf.train.Supervisor(graph=graph, logdir=params.xtep_rnn_model)
    with sv.managed_session() as session:
        if re_train:
            check_point = tf.train.get_checkpoint_state(params.xtep_rnn_model)
            sv.saver.restore(session, check_point.model_checkpoint_path)
        # else:
        #     tf.global_variables_initializer().run()

        for i in range(conf.max_max_epoch):
            lr_decay = conf.lr_decay ** max(i + 1 - conf.max_epoch, 0.0)
            train_model.assign_lr(session, conf.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_model._lr)))
            train_perplexity = train_model.train(session, td)

            print("Epoch: %d Train Perplexity: %.5f" % (i + 1, train_perplexity))
            # valid_perplexity = mvalid.run_epochs(session, valid_data)
            # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            if params.xtep_rnn_model and (i % 10 == 0 or i == conf.max_max_epoch - 1):
                print("Saving model to %s." % params.xtep_rnn_model)
                sv.saver.save(session, params.xtep_rnn_model)

        words = predict_model.run_predict(session, td, wc)
        # print(''.join(wc.ids_to_words(words)).replace('END', '\n'))

