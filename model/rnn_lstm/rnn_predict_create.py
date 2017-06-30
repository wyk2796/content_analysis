# coding:utf-8

import tensorflow as tf
import model.rnn_lstm.rnn_short_text_model as rstm
import model.rnn_lstm.rnn_config as conf
import static_params as params
import wordseg as ws
import model.words_deal as wd
import pre_deal_data.train_data as td
import model.rnn_lstm.rnn_train as tr
from model.rnn_lstm import rnn_train as rt


def create_predict_RNNModel(wc, outpath):

    conf_t = conf.MediumConfig()
    conf_t.vocabulary_size = wc.vocabulary_size
    conf_t.hidden_size = params.embedding_size
    conf_t.num_steps = 30
    conf_t.max_max_epoch = 100
    conf_t.max_epoch = 90
    conf_t.learning_rate = 0.1
    conf_t.lr_decay = 0.9
    conf_t.output_size = wc.label_size()
    conf_t.embedding_init = rt.get_embedding_from_word2vec(params.xtep_words2vec_model)


    predict_conf = conf.MediumConfig()
    predict_conf.vocabulary_size = wc.vocabulary_size
    predict_conf.batch_size = 1
    predict_conf.num_steps = 30
    predict_conf.keep_prob = 1
    predict_conf.output_size = wc.label_size()
    predict_conf.hidden_size = params.embedding_size
    predict_conf.embedding_init = tr.get_embedding_from_word2vec(params.xtep_words2vec_model)
    graph = tf.Graph()

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-predict_conf.init_scale,
                                                    predict_conf.init_scale)

        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                train_model = rstm.RNNModel(conf_t, 'Train')

        with tf.name_scope('Predict'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                predict_model = rstm.RNNModel(predict_conf, 'Predict_argmax')
        input_graph_def = graph.as_graph_def()

    sv = tf.train.Supervisor(graph=graph, logdir=params.xtep_rnn_model)
    with sv.managed_session() as session:
        check_point = tf.train.get_checkpoint_state(params.xtep_rnn_model)
        sv.saver.restore(session, check_point.model_checkpoint_path)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session,
            input_graph_def,
            [predict_model.predict.name.split(':')[0]]
        )

        with tf.gfile.GFile(outpath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    ws_dict = ws.initial_dict(params.dict_path, params.stop_word_path)
    wc = wd.create_content_from_file(params.xtep_words_ids, params.vocabulary_size)
    wc.load__label_word_index(params.label_words_ids)
    create_predict_RNNModel(wc, params.predict_model_constant)

    # train_data = td.TrainData(ws_dict, wc)
    # train_data.add_predict_data_from_file(params.anta_bad_middle_content_path)
    # create_predict_RNNModel(train_data, wc, params.predict_bad_out)
    # train_data.clear_predict()
    # train_data.add_predict_data_from_file(params.anta_good_content_path)
    # create_predict_RNNModel(train_data, wc, params.predict_good_out)












