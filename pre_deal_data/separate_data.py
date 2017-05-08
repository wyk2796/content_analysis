# coding:utf-8

def separate_data_to(data):
    out1 = open('E:\\temp\good_content.txt', encoding='utf-8', mode='w')
    out2 = open('E:\\temp\middle_content.txt', encoding='utf-8', mode='w')
    out3 = open('E:\\temp\\bad_content.txt', encoding='utf-8', mode='w')
    data.apply(func=lambda x: alternate_write(x, out1, out2, out3), axis=1)
    out1.close()
    out2.close()
    out3.close()


def alternate_write(line, out1, out2, out3):
    if str(line['result']) == 'good':
        out1.write('#'.join(map(lambda x: str(x), list(line))) + '\n')
    elif str(line['result']) == 'bad':
        out3.write('#'.join(map(lambda x: str(x), list(line))) + '\n')
    else:
        out2.write('#'.join(map(lambda x: str(x), list(line))) + '\n')