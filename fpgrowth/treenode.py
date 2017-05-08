# coding=utf-8
import time


class treeNode(object):
    """docstring for treeNode"""

    def __init__(self, name_value, num_occur, parent_node):
        super(treeNode, self).__init__()
        self.name = name_value
        self.count = num_occur
        self.nodeLink = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def __lt__(self, other):
        return self.count < other.count

    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createInitSet(dataSet):
    ret_dict = {}
    for trans in dataSet:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


def createTree(dataSet, minSup=1):
    headerTable = {}
    # frequency of each item
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]  # some trans may same
    # remove items not meeting minSup
    nheaader_table = {}
    freq_item_set = set()
    for k in headerTable.keys():
        if headerTable[k] > minSup:
            nheaader_table[k] = [headerTable[k], None]
            freq_item_set.add(k)
    headerTable = nheaader_table
    # freqItemSet = set(headerTable.keys())
    if len(freq_item_set) == 0:  # no frequent item
        return None, None
    # for k in headerTable:  # add a point field
    #     headerTable[k] = [headerTable[k], None]

    ret_tree = treeNode('Null set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:  # 把每一个项集的元素提取出来，并加上统计出来的频率
            if item in freq_item_set:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:  # 排序，并更新树
            orderdItem = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderdItem, ret_tree, headerTable, count)
    return ret_tree, headerTable


def updateTree(items, inTree, headerTable, count):
    # 将新的节点加上来
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新指针
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(node_to_test, target_node):
    while node_to_test.nodeLink is not None:
        node_to_test = node_to_test.nodeLink
        node_to_test.nodeLink = target_node


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # print(len(headerTable))
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]  # (sort header table)
    # print bigL
    for basePat in bigL:  # start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print('finalFrequent Item: ', newFreqSet)  # append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print('condPattBases :', basePat, condPattBases)
        # 2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        # print('head from conditional tree: ', myHead)
        if myHead is not None:  # 3. mine cond. FP-tree
            # print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
