# coding:utf-8

import tensorflow as tf
from model.rnn_lstm.bidirectional import BidirectionalRecurrentN
import static_params as params
import model.rnn_lstm.rnn_train as rt


def train_bidirectional(td, sava_path, re_Train = False):

    graph = tf.Graph()
    batch_size = 20
    hidden_size = params.embedding_size
    keep_prob = 0.5
    num_steps = 30
    embed_init = rt.get_embedding_from_word2vec(params.xtep_words2vec_model)
    classify = {'emotion': 3, 'item_des': 3, 'service_des': 3, 'logistics_des': 3}

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-0.1,
                                                    0.1)

        train_model = BidirectionalRecurrentN()
        train_model.set_batch_size(batch_size)
        train_model.set_hidden_size(hidden_size)
        train_model.set_keep_prob(keep_prob)
        train_model.set_learning_rate(1e-5)
        train_model.set_num_steps(num_steps)
        train_model.get_train_graph(embed_init, params.vocabulary_size, initializer, classify)

        valid_model = BidirectionalRecurrentN()
        valid_model.set_batch_size(batch_size)
        valid_model.set_hidden_size(hidden_size)
        valid_model.set_num_steps(num_steps)
        valid_model.get_valid_graph(embed_init, params.vocabulary_size, initializer, classify)

        predict_model = BidirectionalRecurrentN()
        predict_model.set_batch_size(1)
        predict_model.set_hidden_size(hidden_size)
        predict_model.set_num_steps(num_steps)
        predict_model.get_predict_graph(embed_init, params.vocabulary_size, initializer, classify)

    sv = tf.train.Supervisor(graph=graph)
    with sv.managed_session() as session:

        if re_Train:
            checkpoint = tf.train.get_checkpoint_state(sava_path)
            sv.saver.restore(session, checkpoint.model_checkpoint_path)


        for i in range(50):
            accuracy = dict()
            count = 0
            for step, (input_data, label) in enumerate(td.train_data_label(batch_size, num_steps)):
                acc = train_model.train_run(session, input_data, label)
                for (name, acc_data) in acc.items():
                    accuracy[name] = accuracy.get(name, 0) + acc_data
                count += 1

            for (name, acc_data) in accuracy.items():
                print('epoch %d, classify: %s ,accuracy %f' % (i, name, acc_data / count))
            print('----------------------------------------')

            if i % 10 == 9 and sava_path:
                print('save model to %s' % sava_path)
                sv.saver.save(session, sava_path)

        create_predict_model(predict_model, graph, session, sava_path + 'classify_model.pd')


def create_predict_model(model, graph, session, sava_path):
    predict_name = [pred.name.split(':')[0] for name, pred in model.predict_op.items()]
    print(predict_name)
    input_graph_def = graph.as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        session,
        input_graph_def,
        predict_name
    )
    with tf.gfile.GFile(sava_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())




