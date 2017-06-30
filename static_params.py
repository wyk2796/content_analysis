# coding:utf-8

dict_path = 'E:\词库\contentdict'
dict_file = 'E:\词库\contentdict\dict.txt'

stop_word_path = 'E:\词库\stop_word'
alternate_word_path = 'E:\词库\\alternate_words'

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


words2vec_model_path = 'E:\\temp\\model\\words2vec\\'

#--------------------------------------------------

vocabulary_size = 20000
embedding_size = 500


train_data_o_good = 'E:\\temp\\train_data\\good.txt'
train_data_o_bad = 'E:\\temp\\train_data\\bad.txt'
label_words_ids = 'E:\\temp\model\words\label_words_ids.txt'

predict_model_constant = 'E:\\temp\model\\constant\\rnn_model.pb'
predict_good_out = 'E:\\temp\model\words\good_predict.txt'
predict_bad_out = 'E:\\temp\model\words\\bad_predict.txt'
#---------------------------------------------------

xtep = 'E:\\temp\\xtep-1.csv'
xtep_bad_path = 'E:\\temp\\class_data\\xtep_bad_content.txt'
xtep_good_path = 'E:\\temp\\class_data\\xtep_good_content.txt'
xtep_middle_path = 'E:\\temp\\class_data\\xtep_middle_content.txt'

xtep_bad_middle_content_path = 'E:\\temp\\class_data\\xtep_bad_content_only.txt'
xtep_good_content_path = 'E:\\temp\\class_data\\xtep_good_content_only.txt'

xtep_bad_seg_words_path = 'E:\\temp\\class_data\\xtep_bad_seg_words.txt'
xtep_good_seg_words_path = 'E:\\temp\\class_data\\xtep_good_seg_words.txt'


xtep_words_ids = 'E:\\temp\model\words\words_ids.txt'
xtep_words2vec_model = 'E:\\temp\model\words2vec\\'

xtep_rnn_model = 'E:\\temp\model\\rnn\\'
xtep_rnn_emotion_model = 'E:\\temp\model\\rnn_emotion\\'
xtep_rnn_item_des_model = 'E:\\temp\model\\rnn_item_des\\'
xtep_rnn_service_des_model = 'E:\\temp\model\\rnn_service_des\\'
xtep_rnn_logistics_des_model = 'E:\\temp\model\\rnn_logistics_des\\'

#------------------------------------------------------

anta = 'E:\\temp\\anta_data\\anta.csv'

anta_bad_middel_path = 'E:\\temp\\anta_data\\anta_bad_middle_content.txt'
anta_good_path = 'E:\\temp\\anta_data\\anta_good_content.txt'

anta_bad_middle_content_path = 'E:\\temp\\anta_data\\anta_bad_content_only.txt'
anta_good_content_path = 'E:\\temp\\anta_data\\anta_good_content_only.txt'

anta_bad_seg_words_path = 'E:\\temp\\anta_data\\anta_bad_seg_words.txt'
anta_good_seg_words_path = 'E:\\temp\\anta_data\\anta_good_seg_words.txt'

