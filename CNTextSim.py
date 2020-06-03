# -*- coding:utf-8 -*-

import codecs
import copy
import json
import sys
from datetime import datetime

import jieba
import numpy as np


def get_cos(x, y):
    myx = np.array(x)
    myy = np.array(y)
    cos1 = np.sum(myx * myy)
    cos21 = np.sqrt(sum(myx * myx))
    cos22 = np.sqrt(sum(myy * myy))
    return cos1 / (cos21 * cos22)


if __name__ == '__main__':
    startTime = datetime.now()
    # 读取三个参数
    testFilePath = sys.argv[1]
    sampleFilePath = sys.argv[2]
    resultFilePath = sys.argv[3]
    # 读取样本文件
    f1 = codecs.open(sampleFilePath, "r", "utf-8")
    try:
        f1_text = f1.read()
    finally:
        f1.close()
    # 进行结巴分词
    f1_seg_list = jieba.cut(f1_text)
    # 读取测试文件
    ftest1 = codecs.open(testFilePath, "r", "utf-8")
    try:
        ftest1_text = ftest1.read()
    finally:
        ftest1.close()
    # 进行结巴分词
    ftest1_seg_list = jieba.cut(ftest1_text)

    # 停用词文件路径
    f_stop = codecs.open(r"D:\testfile\stopwords.txt", "r", "utf-8")
    try:
        f_stop_text = f_stop.read()
    finally:
        f_stop.close()
    f_stop_seg_list = f_stop_text.split("\n")

    test_words = {}
    all_words = {}

    for word in f1_seg_list:
        # 去除停用词，读取词集
        if not (word.strip()) in f_stop_seg_list:
            test_words.setdefault(word, 0)
            all_words.setdefault(word, 0)
            all_words[word] += 1

    mytest1_words = copy.deepcopy(test_words)
    for word in ftest1_seg_list:
        # 去除停用词，读取词集
        if not (word.strip()) in f_stop_seg_list:
            if word in mytest1_words:
                mytest1_words[word] += 1

    # 计算相似度
    sampleData = []
    testFileData = []
    for key in all_words.keys():
        sampleData.append(all_words[key])
        testFileData.append(mytest1_words[key])
    testFileSim = get_cos(sampleData, testFileData)

    # 返回结果
    resultJson = json.dumps({
        "test_file_path": testFilePath,
        "sample_file_path": sampleFilePath,
        "result_file_path": resultFilePath,
        "cos_sim_num": testFileSim,
        "begin_time": str(startTime),
        "end_time": str(datetime.now())
    }, ensure_ascii=False)
    print(resultJson)

    resultFile = codecs.open(resultFilePath, "w", "utf-8")
    # 结果写入结果文件
    resultFile.write(resultJson)
    resultFile.close()
