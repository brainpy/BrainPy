# -*- coding: utf-8 -*-

"""
https://blog.csdn.net/xiaodongxiexie/article/details/78836157
"""

def count_code_nums(file):
    with open(file,encoding='utf-8') as data:
        count, flag = 0, 0
        begin = ('"""', "'''")
        for line in data:
            line2 = line.strip()
            if line2.startswith('#'):continue
            elif line2.startswith(begin):
                if line2.endswith(begin) and len(line2) > 3:
                    flag = 0
                    continue
                elif flag == 0:
                    flag = 1
                else:
                    flag = 0
                    continue
            elif flag == 1 and line2.endswith(begin):
                flag = 0
                continue
            if flag == 0 and line2:
                count += 1
    return count


def detect_rows(begin=0, root='.'):
    """
    统计指定文件夹内所有py文件代码量
    :param begin: 起始，一般使用默认0即可
    :param root: 需要统计的文件（文件夹）路径
    :rtype :int
    """
    import os, glob
    for file in glob.glob(os.path.join(root, '*')):
        if os.path.isdir(file):
            begin += detect_rows(0, file)
        elif file.endswith('.py'):
            no = count_code_nums(file)
            print(f"{no}", file)
            begin += no
    return begin


print(detect_rows(root='../npbrain'))
