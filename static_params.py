# coding:utf-8


dict_path = 'E:\MyWork\gitpython\content_analysis\data\dict\contentdict\\'
dict_file = 'E:\MyWork\gitpython\content_analysis\data\dict\contentdict\dict.txt'

stop_word_path = 'E:\MyWork\gitpython\content_analysis\data\dict\stop_word\\'
alternate_word_path = 'E:\MyWork\gitpython\content_analysis\data\dict\\alternate_words'

content_path = 'E:\\temp\\transform_data\\part-00001'

content_bad_path = 'E:\\temp\\class_data\\bad_content.txt'
content_good_path = 'E:\\temp\\class_data\\good_content.txt'
content_middle_path = 'E:\\temp\\class_data\\middle_content.txt'
content_text_path = 'E:\\temp\\class_data\\text.txt'
content_orginal_path = 'E:\\temp\\transform_data'

tf_idf_content = 'E:\\temp\\class_data'
tf_idf_model = 'E:\\temp\\tfidf\\tfidf.txt'

fp_words = 'E:\\temp\\fp_words.txt'

statistic_words = 'E:\\temp\\stat\\words.txt'
statistic_content_num = 'E:\\temp\\stat\\content_num.txt'

table_name = ['tid', 'oid', 'num_iid', 'dp_id', 'valid_score', 'role',
         'nick', 'result', 'created', 'rated_nick', 'item_title',
         'item_price', 'content', 'reply']

middle_table_path = 'E:\\temp\\middle_table'




#--------------------------------------------------

model = 'E:\\temp\model\\'
train_data_dir = 'E:\MyWork\gitpython\content_analysis\data\\train_data'

vocabulary_size = 20000
embedding_size = 500

words2vec_model_path = model + 'words2vec\\'

train_data_o_good = train_data_dir + 'good.txt'
train_data_o_bad = train_data_dir + 'bad.txt'
label_words_ids = model + 'words\label_words_ids.txt'

predict_model_constant = model + 'constant\\rnn_model.pb'
predict_good_out = model + 'words\good_predict.txt'
predict_bad_out = model + 'words\\bad_predict.txt'
#---------------------------------------------------

xtep_dir = 'E:\\temp\\class_data\\'

xtep = xtep_dir + 'xtep-1.csv'
xtep_bad_path = xtep_dir + 'xtep_bad_content.txt'
xtep_good_path = xtep_dir + 'xtep_good_content.txt'
xtep_middle_path = xtep_dir + 'xtep_middle_content.txt'

xtep_bad_middle_content_path = xtep_dir + 'xtep_bad_content_only.txt'
xtep_good_content_path = xtep_dir + 'xtep_good_content_only.txt'

xtep_bad_seg_words_path = xtep_dir + 'xtep_bad_seg_words.txt'
xtep_good_seg_words_path = xtep_dir + 'xtep_good_seg_words.txt'


xtep_words_ids = model + 'words\words_ids.txt'
xtep_words2vec_model = model + 'words2vec\\'

xtep_rnn_model = model + 'rnn\\'
xtep_rnn_seq2seq = model + 'seq2seq\\'
xtep_rnn_emotion_model = model + 'rnn_emotion\\'
xtep_rnn_item_des_model = model + 'rnn_item_des\\'
xtep_rnn_service_des_model = model + 'rnn_service_des\\'
xtep_rnn_logistics_des_model = model + 'rnn_logistics_des\\'

#------------------------------------------------------

anta_dir = 'E:\\temp\\anta_data\\'
anta = anta_dir + 'anta.csv'

anta_bad_middel_path = anta_dir + 'anta_bad_middle_content.txt'
anta_good_path = anta_dir + 'anta_good_content.txt'

anta_bad_middle_content_path = anta_dir + 'anta_bad_content_only.txt'
anta_good_content_path = anta_dir + 'anta_good_content_only.txt'
anta_bad_middle_content_split_path = anta_dir + 'anta_bad_content_split.txt'

anta_bad_seg_words_path = anta_dir + 'anta_bad_seg_words.txt'
anta_good_seg_words_path = anta_dir + 'anta_good_seg_words.txt'
anta_predict_good = anta_dir + 'predict_good.txt'

