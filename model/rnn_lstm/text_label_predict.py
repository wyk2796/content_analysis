# coding:utf-8

import tensorflow as tf
import static_params as params
import wordseg as ws
import model.words_deal as wd
import pre_deal_data.train_data as td

class Predict(object):

    def __init__(self, model_path, model_name='default'):
        self.model_path = model_path
        self.model_name = model_name
        self.feed = dict()
        self.fetch = dict()
        self.graph = self.load_graph()
        self.session = tf.Session(graph=self.graph)

    def add_fetch(self, outputs):
        for (class_name, output_name) in outputs.items():
            self.fetch[class_name] = self.graph.get_tensor_by_name(self._get_tensor_name(output_name))

    def add_feed(self, inputs):
        for (class_name, inputs_name) in inputs.items():
            self.feed[class_name] = self.graph.get_tensor_by_name(self._get_tensor_name(inputs_name))

    def session_run(self, data_input):
        self.session.run(self.feed, self.fetch)

    def load_graph(self):
        with tf.gfile.GFile(self.model_path, 'rb') as g:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(g.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.model_name)
        return graph

    def _get_tensor_name(self, name):
        return '%s/%s:0' % (self.model_name, name)



def load_graph(model_file_path, name='default'):
    with tf.gfile.GFile(model_file_path, 'rb') as g:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(g.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=name)
    return graph


def predict_text_label(model_file_path, output_path, pd, wc, num_steps=30):
    graph = load_graph(model_file_path)
    for op in graph.get_operations():
        print(op.name, op.values())

    initial_c1 = graph.get_tensor_by_name('default/Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0')
    initial_h1 = graph.get_tensor_by_name('default/Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0')
    initial_c2 = graph.get_tensor_by_name('default/Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0')
    initial_h2 = graph.get_tensor_by_name('default/Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0')

    finial_state_c1 = graph.get_tensor_by_name('default/Predict/Model/RNN/multi_rnn_cell_29/cell_0/basic_lstm_cell/add_1:0')
    finial_state_h1 = graph.get_tensor_by_name('default/Predict/Model/RNN/multi_rnn_cell_29/cell_0/basic_lstm_cell/mul_2:0')
    finial_state_c2 = graph.get_tensor_by_name('default/Predict/Model/RNN/multi_rnn_cell_29/cell_1/basic_lstm_cell/add_1:0')
    finial_state_h2 = graph.get_tensor_by_name('default/Predict/Model/RNN/multi_rnn_cell_29/cell_1/basic_lstm_cell/mul_2:0')

    input_data = graph.get_tensor_by_name('default/Predict/Model/RNN/Input:0')
    predict_out = graph.get_tensor_by_name('default/Predict/Model/predict_out:0')

    with tf.Session(graph=graph) as session:
        initial_state = ((initial_c1, initial_h1), (initial_c2, initial_h2))
        finial_state = ((finial_state_c1, finial_state_h1), (finial_state_c2, finial_state_h2))
        i_c1, i_h1, i_c2, i_h2 = session.run([initial_c1, initial_h1, initial_c2, initial_h2])
        state = ((i_c1, i_h1), (i_c2, i_h2))

        fetches = {
            "predict": predict_out,
            "final_state": finial_state
        }
        o = open(output_path, encoding='utf-8', mode='w')
        train_input, text = pd.predict_data_generate(num_steps)
        while train_input is not None:
            feed = {input_data: [train_input],
                    initial_state: state}
            vals = session.run(
                fetches, feed_dict=feed)
            y_p = vals["predict"]
            y_predict = [w for w in list(y_p) if w != 0]
            state = vals["final_state"]
            predict_words = ''.join(wc.label_ids_to_words(y_predict))
            o.write('[%s]\t%s\n' % (predict_words.replace('END', ','), text))
            train_input, text = pd.predict_data_generate(num_steps)
        o.close()


def predict_label(session, graph, pd, graph_name, input):
    input_data = graph.get_tensor_by_name(graph_name + '/Predict_1/model/text_input')
    emotion_output = graph.get_tensor_by_name(graph_name + '/Predict_1/model/emotion_output')
    item_des_output = graph.get_tensor_by_name(graph_name + 'Predict_1/model/item_des_output')
    service_des_output = graph.get_tensor_by_name(graph_name + 'Predict_1/model/service_des_output')
    logistics_des_output = graph.get_tensor_by_name(graph_name + 'Predict_1/model/logistics_des_output')

    fetches = {
        'emotion_class': emotion_output,
        'item_des_class': item_des_output,
        'service_des_class': service_des_output,
        'logistics_des_class': logistics_des_output
    }

    feed = {
        input_data: input
    }
    result = session.run(fetches, feed)

    return result





def predict_text_and_label(text_model_file_path, classify_model_file, output_path, pd, wc, num_steps=30):
    text_model = load_graph(text_model_file_path)
    classify_model = load_graph(classify_model_file)


if __name__ == '__main__':
    ws_dict = ws.initial_dict(params.dict_path, params.stop_word_path)
    wc = wd.create_content_from_file(params.xtep_words_ids, params.vocabulary_size)
    wc.load__label_word_index(params.label_words_ids)
    train_data = td.TrainData(ws_dict, wc)
    train_data.add_predict_data_from_file(params.anta_bad_middle_content_path)
    predict_text_label(params.predict_model_constant, params.predict_bad_out, train_data, wc)
