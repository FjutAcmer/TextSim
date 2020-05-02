# -*- coding:utf-8 -*-

import codecs
import copy
import sys
import json
import time
from datetime import date

import jieba
import numpy as np


def get_cos(x, y):
    myx = np.array(x)
    myy = np.array(y)
    cos1 = np.sum(myx * myy)
    cos21 = np.sqrt(sum(myx * myx))
    cos22 = np.sqrt(sum(myy * myy))
    return cos1 / (cos21 * cos22)


startTime = time.time()
testFilePath = sys.argv[1]
sampleFilePath = sys.argv[2]
resultFilePath = sys.argv[3]

f1 = codecs.open(sampleFilePath, "r", "utf-8")
try:
    f1_text = f1.read()
finally:
    f1.close()

f1_seg_list = jieba.cut(f1_text)
# first test
ftest1 = codecs.open(testFilePath, "r", "utf-8")
try:
    ftest1_text = ftest1.read()
finally:
    ftest1.close()
ftest1_seg_list = jieba.cut(ftest1_text)

# read sample text
# remove stop word and constructor dict
f_stop = codecs.open(r"D:\testfile\stopwords.txt", "r", "utf-8")
try:
    f_stop_text = f_stop.read()
finally:
    f_stop.close()
f_stop_seg_list = f_stop_text.split("\n")

test_words = {}
all_words = {}

for word in f1_seg_list:
    if not (word.strip()) in f_stop_seg_list:
        test_words.setdefault(word, 0)
        all_words.setdefault(word, 0)
        all_words[word] += 1

    # read to be tested word
mytest1_words = copy.deepcopy(test_words)
for word in ftest1_seg_list:
    if not (word.strip()) in f_stop_seg_list:
        if word in mytest1_words:
            mytest1_words[word] += 1

# calculate sample with to be tested text sample
sampleData = []
testFileData = []
for key in all_words.keys():
    sampleData.append(all_words[key])
    testFileData.append(mytest1_words[key])
testFileSim = get_cos(sampleData, testFileData)

resultJson = json.dumps({
    "测试文件路径": testFilePath,
    "样本文件路径": sampleFilePath,
    "结果文件路径": resultFilePath,
    "余弦相似度": testFileSim,
    "测试开始时间": startTime,
    "测试结束时间": time.time()
}, ensure_ascii=False)

print(resultJson)
resultFile = codecs.open(resultFilePath, "w", "utf-8")
resultFile.write(resultJson)
resultFile.close()
