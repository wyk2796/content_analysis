# coding:utf-8
import static_params as params


def resutl_deal(path):
    with open(path, encoding='utf-8', mode='r') as file:
        data = file.readlines()
        content_map = dict()
        for line in data:
            elem = line.strip().split('\t')
            if elem[0] not in content_map:
                if len(elem) == 6:
                    content_map[elem[0]] = (elem[1], elem[2], elem[3], elem[4], elem[5])
                if len(elem) == 5:
                    content_map[elem[0]] = (elem[1], elem[2], elem[3], elem[4], '')
            if elem[0] in content_map:
                (e, i, s, l, la) = content_map[elem[0]]
                if int(e) > int(elem[1]):
                    e = elem[1]
                if int(i) < int(elem[2]):
                    i = elem[2]
                if int(s) < int(elem[3]):
                    s = elem[3]
                if int(l) < int(elem[4]):
                    l = elem[4]
                label = []
                label.extend(la.split(','))
                if len(elem) == 6:
                    label.extend(elem[5].split(','))
                la = ','.join(set(label))
                content_map[elem[0]] = (e, i, s, l, la)
        con = sorted(content_map.items(), key=lambda x: len(x[0]))
    with open(path, encoding='utf-8', mode='w') as file:
        for sen, (e, i, s, l, la) in con:
            ssr = '%s\t%s\t%s\t%s\t%s\t%s\n' % (sen, e, i, s, l, la)
            file.write(ssr)


def split_line(in_path, out_path):
    in_stream = open(in_path, encoding='utf-8', mode='r')
    out_stream = open(out_path, encoding='utf-8', mode='w')
    for line in in_stream.readlines():
        for sen in line.strip().split(','):
            if sen.isalnum() and not sen.isdigit() and not sen.isspace():
                out_stream.write(sen.strip() + '\n')
    in_stream.close()
    out_stream.close()


if __name__ == '__main__':
    resutl_deal(params.predict_good_out)
    #split_line(params.anta_bad_middle_content_path, )


