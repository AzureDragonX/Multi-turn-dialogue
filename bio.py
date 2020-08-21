# -*- coding:utf-8 -*-
#将文本处理为BIO标注
#2019/6/20

# __author__  = "ljq"
import os
import re
import codecs


def getCode(expr, tagsDict):
    """
    根据标注名称返回标注
    :param expr:
    :param tagsDict: 标注字典映射
    :return:
    """
    tagsNameDict = {value:key for key, value in tagsDict.items()}
    if expr in tagsNameDict:
        return tagsNameDict[expr]
    else:
        return ""

def createMapping(filePath, fileName, tagsDict):
    file = os.path.join(filePath, fileName)
    mapping = dict()
    fAnn = open(file, 'r', encoding= 'utf-8') #打开文件

    for line in fAnn.readlines():
        #分割  
        line = re.split(r'[\s]',line) #['T3', '临床表现', '15', '17', '咳嗽', '']
   
        #获取 3种关系，转化为BIO标注
        expression = line[1] #'临床表现'
        #获取实体
        word = " ".join(line[4:]).strip(" ") #'咳嗽'
        #获取实体的下标
        label = line[2] #'15'
        #定义一个实体获取函数
        code = getCode(expression, tagsDict) #'CF'
        for i, char in enumerate(word):
            value = ""
            number = int(label) #15

            if i == 0:
                char = char + label #咳15
                #print(char)
                value = "B-" + code #'B-CF'
        
            else:
                label = int(label) #15
                id= label + i
                id = str(id)
                char = char + id #'嗽16'
                value = "I-" + code #'I-CF'
            mapping[char] = value #{'咳15': 'B-CF', '嗽16': 'I-CF'}
    return mapping


def toBioLine(mapping, filePath, filName, desPath):
    """
        将*.txt转换为*.bio，为每个txt中的字符，找到其对应的bio标注符。
    Args:
    @mapping 实体-标注词典
    @filName *.txt文件名

    """
    
    i = 0
    spt = filName.split('.')
    spt[0] = spt[0] + '.out'
    #destFileName = ''.join(spt)
    destFileName = spt[0]
    source = os.path.join(filePath, filName)#原始文本txt
    f1 = open(source, 'r', encoding= 'utf-8')

    dest = os.path.join(desPath, destFileName)
    outputData = codecs.open(dest, 'w',  encoding= 'utf-8') #'.\\modelFile\\traindata_600\\376.out'
    for line in f1.readlines(): # line是原文txt
        for word in line:
            i = i + 1
            id = str(i-1)
            wordId = word + id
            if mapping.get(wordId) == None:#找到映射表中对应的词
                if word == '\n':

                # print(word)
                    outputData.write(' ')
                elif word == ' ':
                    outputData.write(word +" "+ " O")
                elif word == '\u3000':
                    outputData.write(word +" "+ " O")
                else:
                    outputData.write(word + " O")
            else :
                outputData.write(word + " " + mapping[wordId])
            outputData.write("\n")
    outputData.close()
    #print("success!")