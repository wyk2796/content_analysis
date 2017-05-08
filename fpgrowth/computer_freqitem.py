# coding:utf-8

from fpgrowth import treenode as tn


def computer_apri(data, min_sup1, min_sup2):
    init_set = tn.createInitSet(data)
    my_fp, my_head_table = tn.createTree(init_set, min_sup1)
    # my_fp.disp()
    freq_items = []
    tn.mineTree(my_fp, my_head_table, min_sup2, set([]), freq_items)
    return freq_items


