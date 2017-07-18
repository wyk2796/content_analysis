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
        self.tensor = dict()
        self.graph = self.load_graph()
        self.session = tf.Session(graph=self.graph)

    def add_tensor(self, tensor_map):
        for (class_name, tensor_name) in tensor_map.items():
            self.tensor[class_name] = self.graph.get_tensor_by_name(self._get_tensor_name(tensor_name))

    def session_run(self, data_input, outputs):
        feed = dict()
        fetch = dict()
        for (name, data) in data_input.items():
            feed[self.tensor[name]] = data

        for out in outputs:
            fetch[out] = self.tensor[out]
        return self.session.run(fetch, feed)

    def load_graph(self):
        with tf.gfile.GFile(self.model_path, 'rb') as g:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(g.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.model_name)
        return graph

    def _get_tensor_name(self, name):
        return '%s/%s:0' % (self.model_name, name)

    def close(self):
        self.session.close()


def predict_text_and_label(text_model_file_path, classify_model_file, output_path, pd, wc, num_steps=30):
    classify_model = Predict(classify_model_file, 'classify')
    classify_tensor = {
        'class_input': 'Predict_1/model/text_input',
        'emotion_class': 'Predict_1/model/emotion_output',
        'item_des_class': 'Predict_1/model/item_des_output',
        'service_des_class': 'Predict_1/model/service_des_output',
        'logistics_des_class': 'Predict_1/model/logistics_des_output'
    }
    classify_model.add_tensor(classify_tensor)

    text_label_model = Predict(text_model_file_path, 'text_label')
    text_label_tensor = {
        'input_data': 'Predict/Model/RNN/Input',
        'initial_c1': 'Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros',
        'initial_h1': 'Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1',
        'initial_c2': 'Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros',
        'initial_h2': 'Predict/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1',
        'predict_out': 'Predict/Model/predict_out',
        'finial_state_c1': 'Predict/Model/RNN/multi_rnn_cell_29/cell_0/basic_lstm_cell/add_1',
        'finial_state_h1': 'Predict/Model/RNN/multi_rnn_cell_29/cell_0/basic_lstm_cell/mul_2',
        'finial_state_c2': 'Predict/Model/RNN/multi_rnn_cell_29/cell_1/basic_lstm_cell/add_1',
        'finial_state_h2': 'Predict/Model/RNN/multi_rnn_cell_29/cell_1/basic_lstm_cell/mul_2'
    }
    text_label_model.add_tensor(text_label_tensor)

    file_out = open(output_path, mode='w', encoding='utf-8')
    train_input, text = pd.predict_data_generate(num_steps)
    feed_text_label = text_label_model.session_run(dict(), ['initial_c1', 'initial_h1', 'initial_c2', 'initial_h2'])
    feed_classify = dict()
    out_text_label = ['predict_out', 'finial_state_c1', 'finial_state_h1', 'finial_state_c2', 'finial_state_h2']
    out_classify = ['emotion_class', 'item_des_class', 'service_des_class', 'logistics_des_class']

    while train_input is not None:

        feed_text_label['input_data'] = [train_input]
        feed_classify['class_input'] = [train_input]

        result_text_label = text_label_model.session_run(feed_text_label, out_text_label)

        feed_text_label['initial_c1'] = result_text_label['finial_state_c1']
        feed_text_label['initial_c2'] = result_text_label['finial_state_c2']
        feed_text_label['initial_h1'] = result_text_label['finial_state_h1']
        feed_text_label['initial_h2'] = result_text_label['finial_state_h2']

        result_classify = classify_model.session_run(feed_classify, out_classify)
        y_predict = [w for w in list(result_text_label['predict_out']) if w != 0]
        label_words = ''.join(wc.label_ids_to_words(y_predict)).replace('END', ',')

        out_str = '%s\t%d\t%d\t%d\t%d\t%s\n' % (text,
                                                result_classify['emotion_class'],
                                                result_classify['item_des_class'],
                                                result_classify['service_des_class'],
                                                result_classify['logistics_des_class'],
                                                label_words)
        file_out.write(out_str)
        train_input, text = pd.predict_data_generate(num_steps)
    text_label_model.close()
    classify_model.close()
    file_out.close()


if __name__ == '__main__':
    ws_dict = ws.initial_dict(params.dict_path, params.stop_word_path)
    wc = wd.create_content_from_file(params.xtep_words_ids, params.vocabulary_size)
    wc.load__label_word_index(params.label_words_ids)
    train_data = td.TrainData(ws_dict, wc)
    train_data.add_predict_data_from_file(params.anta_predict_good)

    predict_text_and_label(params.predict_model_constant,
                           params.xtep_rnn_emotion_model + '\\classify_model.pd',
                           params.predict_good_out, train_data, wc, num_steps=35)

