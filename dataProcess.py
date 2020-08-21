# -*- coding:utf-8 -*-
'''
@Author: your name
@Date: 2020-01-18 16:09:41
@LastEditTime : 2020-01-18 17:55:58
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \BiLstmCrf_import\dataProcess.py
'''
import re
import pandas as pd

import os
import sys
import importlib
BaseDir = os.path.dirname(os.path.abspath(__file__))
if "afw_ai_engine" in BaseDir:
    BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai_engine.",1)[-1]
else:
    BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai.",1)[-1]
# bio = importlib.import_module(BaseImportDir+".bio")
import bio as bio
getCode = bio.getCode
createMapping = bio.createMapping
toBioLine = bio.toBioLine

class StandardError(Exception):
    def __init__(self, err):
        Exception.__init__(self, err)

def dataProcess(modelPath, tagsDict):
        #trainPath = ".\BIO-ALL格式/600" #文件夹目录
        trainPath =modelPath
        files= os.listdir(trainPath) #得到文件夹下的所有文件名称
        s = []
        prefixs = map(lambda x:  x.split('.')[0], files)
        prefixs = list(set(prefixs))           
        # try:
        for prefix in prefixs:
                #遍历文件夹
                annFile = prefix + ".ann"
                txtFile = prefix + ".txt"
                mapping = createMapping(trainPath, annFile, tagsDict)
                toBioLine(mapping, trainPath, txtFile, trainPath)
        # except Exception as e:
        #         raise StandardError("输出单个bio文件错误！！！")
        try:
                txtPath=trainPath
                nameList=[x for x in os.listdir(txtPath) ] #列出目录所有文件名
                nameListAnn = [] #装所有.out文件
                for i in range(len(nameList)): #找出所有.out文件
                        if re.search(r'.out',nameList[i]):
                                nameList1 = nameList[i]
                                nameListAnn.append(nameList1)
                outFileName='bio_all.txt'
                outputList = []
                outFile=open(outFileName,'w',encoding='utf-8')#a没有文件可以创建
                for i in range(len(nameListAnn)):#解析.out文件格式，提取成[['患', 'O'],..]形式的列表数据
                        dataPath = os.path.join(txtPath, nameListAnn[i])
                        f=open(dataPath,'r', encoding='utf-8')
                        for line in f.readlines():#读取.out文件的每一行
                                line = line.strip('\n')
                                if line[0] == "\u3000":
                                        continue
                                lineTmp = line.split(' ')#原始只有一个空格分割后是两个空值,原始有两个空格分割后是三个空值
                                if len(lineTmp) != 2:
                                    if lineTmp[-1] != '':
                                        line = [' ', lineTmp[-1]]
                                    else:
                                        line = ['', '']
                                else:
                                    if "" == lineTmp[0] and "" ==lineTmp[1]:
                                        line = ['', '']
                                    else:
                                        line = lineTmp
                                outputList.append(line)
        except Exception as e:
                raise StandardError("汇总bio数据出错！！！") 
        try:
                df = pd.DataFrame(outputList, columns=['srcChar', 'NERLabel'])
                return df
        except Exception as e:
                raise StandardError("生成训练、测试、验证数据出错！！！") 