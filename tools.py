# coding:utf-8


def dict_write_to_file(path, dict_list):
    if len(dict_list) > 0 and isinstance(dict_list, dict):
        with open(path, encoding='utf-8', mode='w') as out:
            for line in dict_list.items():
                try:
                    out.write('%s\t%f\n' % (line[0], line[1]))
                except Exception as e:
                    print('read file: %s encounter an error %s' % (path, e))

