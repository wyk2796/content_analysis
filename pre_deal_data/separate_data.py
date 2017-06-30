# coding:utf-8
from pre_deal_data import data_deal as dd


def separate_data_to(data, fun, path1, path2, path3, path4):
    out1 = open(path1, encoding='utf-8', mode='w')
    out2 = open(path2, encoding='utf-8', mode='w')
    out3 = open(path3, encoding='utf-8', mode='w')
    out4 = open(path4, encoding='utf-8', mode='w')
    [alternate_write(line, fun, out1, out2, out3, out4) for line in data]
    out1.close()
    out2.close()
    out3.close()
    out4.close()


def alternate_write(line, fun, out1, out2, out3, out4):
    try:
        if fun(line) == 'good' and line[-2] != '好评！' and pre_line(line[-2]):
            out1.write(','.join(map(lambda x: str(x), list(line))) + '\n')
            out2.write(dd.change_sign_in_content(line[-2]) + '\n')
        elif fun(line) == 'bad' and pre_line(line[-3]):
            out3.write(','.join(map(lambda x: str(x), list(line))) + '\n')
            out4.write(dd.change_sign_in_content(line[-2]) + '\n')
    except Exception as e:
        print('separate encounter an error', e, line)


def pre_line(line):
    line_len = len(line)
    char_len = len(set(line))
    if line_len < 4 or char_len < 4 or char_len / line_len < 0.5:
        return False
    else:
        return True

