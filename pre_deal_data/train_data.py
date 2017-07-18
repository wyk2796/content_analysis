# coding:utf-8
import static_params as params
import numpy as np
import wordseg as ws
import model.words_deal as wd


class TrainData(object):

    def __init__(self, word_seg, word_content):
        self.word_seg = word_seg
        self.word_content = word_content
        self.label_map = dict()
        self.valid_map = dict()
        self.predict_data = list()
        self.predict_seg_data = list()
        self.predict_text = list()

    def size(self):
        return len(self.label_map)

    def clear_predict(self):
        self.predict_data = list()

    def clear_label_map(self):
        self.label_map = dict()

    def add_train_data_from_file(self, path):
        self._add_content_file_to_map(path, self.label_map)
        return self

    def add_valid_data_from_file(self, path):
        self._add_content_file_to_map(path, self.valid_map)

    def _add_content_file_to_map(self, path, label_map):
        with open(path, encoding='utf-8', mode='r') as i_stream:
            data = i_stream.readlines()
            del data[0]
            for line in data:
                content = line.strip().split(sep='\t')
                if content != '' and content[0] not in label_map:
                    try:
                        if len(content) == 6 and content[0] not in label_map:
                            label_map[content[0]] = ContentLabel(content[1],
                                                                      content[2],
                                                                      content[3],
                                                                      content[4],
                                                                      content[5])
                        elif len(content) == 5 and content[0] not in label_map:
                            label_map[content[0]] = ContentLabel(content[1],
                                                                      content[2],
                                                                      content[3],
                                                                      content[4],
                                                                      '')

                    except Exception as e:
                        print('line %s format error %s' % ('\t'.join(content), e))
        return self

    def add_predict_data_from_file(self, path):
        with open(path, encoding='utf-8', mode='r') as i_stream:
            [self.predict_data.append(line.strip()) for line in i_stream.readlines()]

    def get_content_label(self):
        return [v.content_label for (k, v) in self.label_map.items()]

    def print(self):
        [print(k, v.print_str()) for (k, v) in self.label_map.items()]

    def train_data_content(self, batch_size, step_num):
        epoch_size = self.size() // batch_size
        feature = np.zeros((epoch_size * batch_size, step_num), dtype=np.int32)
        labels = np.zeros((epoch_size * batch_size, step_num), dtype=np.int32)
        indexes = np.random.choice(self.size(), epoch_size * batch_size, replace=False)
        data = list(self.label_map.items())
        for step, j in enumerate(indexes):
            (k, v) = data[j]
            lv = v.content_label
            lv = lv.replace(',', wc.interval_char)
            content = np.zeros(step_num, dtype=np.int32)
            labeled = np.zeros(step_num, dtype=np.int32)
            content_words = self.word_content.words_to_ids(self.word_seg.word_cut_with_sign(k))
            ll_w = [wc.start_char]
            ll_w.extend(self.word_seg.word_cut_with_sign(lv))
            ll_w.append(wc.end_char)
            label_words = self.word_content.label_words_to_ids(ll_w)
            if len(content_words) < 35:
                for k in range(len(content_words)):
                    content[k] = content_words[k]
                for t in range(len(label_words)):
                    labeled[t] = label_words[t]
            feature[step] = content
            labels[step] = labeled
        for i in range(epoch_size):
            x = feature[batch_size * i: batch_size * (i + 1)]
            y = labels[batch_size * i: batch_size * (i + 1)]
            yield x, y

    def train_data_attention_content(self, batch_size, step_num):
        epoch_size = self.size() // batch_size
        feature = np.zeros((epoch_size * batch_size, step_num), dtype=np.int32)
        labels_decode = np.zeros((epoch_size * batch_size, step_num), dtype=np.int32)
        target = np.zeros((epoch_size * batch_size, step_num), dtype=np.int32)
        indexes = np.random.choice(self.size(), epoch_size * batch_size, replace=False)
        data = list(self.label_map.items())
        for step, j in enumerate(indexes):
            (k, v) = data[j]
            lv = v.content_label
            lv = lv.replace(',', wc.interval_char)
            content = np.zeros(step_num, dtype=np.int32)
            labeled = np.zeros(step_num, dtype=np.int32)
            content_words = self.word_content.words_to_ids(self.word_seg.word_cut_with_sign(k))
            ll_w = [wc.start_char]
            ll_w.extend(self.word_seg.word_cut_with_sign(lv))
            ll_w.append(wc.end_char)
            label_words = self.word_content.label_words_to_ids(ll_w)
            if len(content_words) < 35:
                for k in range(len(content_words)):
                    content[k] = content_words[k]
                for t in range(len(label_words)):
                    labeled[t] = label_words[t]
            feature[step] = content
            labels_decode[step] = labeled
            t_list = list(labeled[1:])
            t_list.append(0)
            target[step] = np.array(t_list, dtype=np.int32)
        for i in range(epoch_size):
            x = feature[batch_size * i: batch_size * (i + 1)]
            d = labels_decode[batch_size * i: batch_size * (i + 1)]
            y = target[batch_size * i: batch_size * (i + 1)]
            yield x, d, y

    def train_data_label(self, batch_size, step_num):
        epoch_size = self.size() // batch_size
        feature = np.zeros((epoch_size * batch_size, step_num), dtype=np.int32)
        labels = []
        indexes = np.random.choice(self.size(), epoch_size * batch_size, replace=False)
        data = list(self.label_map.items())
        for step, j in enumerate(indexes):
            (k, v) = data[j]
            le = self._generate_one_hot(v.emotion, 3)
            li = self._generate_one_hot(v.item_des, 3)
            ls = self._generate_one_hot(v.service_des, 3)
            ll = self._generate_one_hot(v.logistics_des, 3)
            content = np.zeros(step_num, dtype=np.int32)
            content_words = self.word_content.words_to_ids(self.word_seg.word_cut_with_sign(k))
            for k in range(len(content_words)):
                content[k] = content_words[k]
            feature[step] = content
            labels.append({'emotion': le, 'item_des': li, 'service_des': ls, 'logistics_des': ll})
        for i in range(epoch_size):
            y_label = {'emotion': np.zeros((batch_size, 3), dtype=np.int32),
                       'item_des': np.zeros((batch_size, 3), dtype=np.int32),
                       'service_des': np.zeros((batch_size, 3), dtype=np.int32),
                       'logistics_des': np.zeros((batch_size, 3), dtype=np.int32)}
            x = feature[batch_size * i: batch_size * (i + 1)]
            y = labels[batch_size * i: batch_size * (i + 1)]
            for line_index in range(len(y)):
                for (name, l) in y[line_index].items():
                    y_label[name][line_index] = l
            yield x, y_label

    def _generate_one_hot(self, value, output_size):
        label = np.zeros(output_size, dtype=np.int32)
        label[value] = 1
        return label

    def predict_data_generate(self, step_num):
        def get_data():
            seg_data = self.predict_seg_data[0]
            text = self.predict_text[0]
            del self.predict_seg_data[0]
            del self.predict_text[0]
            return seg_data, text

        if len(self.predict_seg_data) > 0:
            return get_data()
        elif len(self.predict_data) > 0:
            while len(self.predict_seg_data) == 0 and len(self.predict_data) > 0:
                self.predict_data_seg(step_num)
            return get_data()
        else:
            return None, None

    def predict_data_seg(self, step_num):
        if len(self.predict_data) > 0:
            line = self.predict_data[0]
            content_words = self.word_content.words_to_ids(self.word_seg.word_cut_with_sign(line))
            batch = (len(content_words) // step_num) + 1
            batch_len = len(content_words) // batch
            for i in range(batch):
                content = np.zeros(step_num, dtype=np.int32)
                for j in range(batch_len):
                    content[j] = content_words[batch_len * i + j]
                self.predict_seg_data.append(content)
                self.predict_text.append(line)
        del self.predict_data[0]

    def generate_valid_data(self, step_num):
        data = list(self.valid_map.items())
        feature = []
        labels = []
        for (key, labels) in data:
            lv = labels.content_label
            lv = lv.replace(',', wc.interval_char)
            content = np.zeros(step_num, dtype=np.int32)
            labeled = np.zeros(step_num, dtype=np.int32)
            content_words = self.word_content.words_to_ids(self.word_seg.word_cut_with_sign(key))
            ll_w = [wc.start_char]
            ll_w.extend(self.word_seg.word_cut_with_sign(lv))
            ll_w.append(wc.end_char)
            ll_w = [wc.start_char]
            label_words = self.word_content.label_words_to_ids(ll_w)

            if len(content_words) < 35:
                for k in range(len(content_words)):
                    content[k] = content_words[k]
                for t in range(len(label_words)):
                    labeled[t] = label_words[t]
                feature.append(content)
                labels.append(labeled)
        for i in range(len(feature)):
            x = feature[i]
            y = labels[i]
            yield x, y

    def valid_data_label(self, step_num):
        feature = []
        labels = []
        data = list(self.valid_map.items())
        for (key, v) in data:
            le = self._generate_one_hot(v.emotion, 3)
            li = self._generate_one_hot(v.item_des, 3)
            ls = self._generate_one_hot(v.service_des, 3)
            ll = self._generate_one_hot(v.logistics_des, 3)
            content = np.zeros(step_num, dtype=np.int32)
            content_words = self.word_content.words_to_ids(self.word_seg.word_cut_with_sign(key))
            for k in range(len(content_words)):
                content[k] = content_words[k]
            feature.append(content)
            labels.append({'emotion': le, 'item_des': li, 'service_des': ls, 'logistics_des': ll})
        for i in range(len(feature)):
            y_label = {'emotion': np.zeros(3, dtype=np.int32),
                       'item_des': np.zeros(3, dtype=np.int32),
                       'service_des': np.zeros(3, dtype=np.int32),
                       'logistics_des': np.zeros(3, dtype=np.int32)}
            x = feature[i]
            y = labels[i]
            for (name, l) in y.items():
                y_label[name] = l
            yield x, y_label


class ContentLabel(object):

    def __init__(self, emotion, item_des, service_des, logistics_des, content_label):
        self.emotion = int(emotion)
        self.item_des = int(item_des)
        self.service_des = int(service_des)
        self.logistics_des = int(logistics_des)
        self.content_label = content_label

    def print_str(self):
        return ('labels emotion:%d, item_des:%d, service_des:%d, logistics_des:%d, content_label:%s' %
                (self.emotion, self.item_des, self.service_des, self.logistics_des, self.content_label))

import model.rnn_lstm.rnn_train as rt
import model.rnn_lstm.train_bidirectional as tb
if __name__ == '__main__':
    ws_dict = ws.initial_dict(params.dict_path, params.stop_word_path)
    # ws_dict.del_word_in_dict('价格合理')
    # ws_dict.del_word_in_dict('价格便宜')
    # ws_dict.del_word_in_dict('不透气')
    # ws_dict.adjust_freq(('价格', '合理'))
    # ws_dict.adjust_freq(('价格', '便宜'))
    # ws_dict.adjust_freq(('不', '透气'))
    wc = wd.create_content_from_file(params.xtep_words_ids, params.vocabulary_size)
    wc.load__label_word_index(params.label_words_ids)
    train_data = TrainData(ws_dict, wc)
    train_data.add_train_data_from_file(params.train_data_o_good)
    train_data.add_train_data_from_file(params.train_data_o_bad)
    # label_data = [ws_dict.word_cut_with_sign(line) for line in train_data.get_content_label()]
    # wc.create_label_content(label_data)
    # wc.save_label_word_index(params.label_words_ids)
    # rt.RNN_train_2(train_data, wc, re_train=True)
    # for i, (train, label) in enumerate(train_data.train_data_content(20, 30)):
    #     print(i)
    #     print('train', train)
    #     print('label', label)
    # for i, (train, label) in enumerate(train_data.train_data_label(20, 30, 'item_des')):
    #     print(i)
    #     print('train', train)
    #     print('label', label)
    #tb.train_bidirectional(train_data, params.xtep_rnn_emotion_model, re_Train=True)
    rt.RNN_train_seq2seq(train_data, wc, re_train=True)

