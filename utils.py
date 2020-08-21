#-*- coding:utf-8 -*-
import os
import json
import shutil
import logging

import tensorflow as tf

import sys
import importlib
BaseDir = os.path.dirname(os.path.abspath(__file__))
if "afw_ai_engine" in BaseDir:
    BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai_engine.",1)[-1]
else:
    BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai.",1)[-1]
# conlleval = importlib.import_module(BaseImportDir+".conlleval")
import conlleval as conlleval
returnReport = conlleval.returnReport

modelsPath = "./models"
evalPath = "./evaluation"
evalTemp = os.path.join(evalPath, "tmp")
evalScript = os.path.join(evalPath, "conlleval")

#
# def getLogger(logFile):
#     logger = logging.getLogger(logFile)
#     # logger.setLevel(logging.DEBUG)
#     fh = logging.FileHandler(logFile)
#     fh.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     ch.setFormatter(formatter)
#     fh.setFormatter(formatter)
#     logger.addHandler(ch)
#     logger.addHandler(fh)
#     return logger

def copyFile(sourcePath, targetPath):

    source = os.path.abspath(sourcePath)
    target = os.path.abspath(targetPath)

    if not os.path.exists(target):
        os.makedirs(target)

    if os.path.exists(source):
        for root, dirs, files in os.walk(source):
            for file in files:
                srcFile = os.path.join(root, file)
                shutil.copy(srcFile, target)

def testNer(results, path):
    """
    Run perl script to evaluate model
    """
    outputFile = os.path.join(path, "nerPredict.utf8")
    with open(outputFile, "w", encoding='utf-8') as f:
        toWrite = []
        for block in results:
            for line in block:
                toWrite.append(line + "\n")
            toWrite.append("\n")
        f.writelines(toWrite)

    evalResult, evalResultDict = returnReport(outputFile)
    return evalResult, evalResultDict

def stdoutTrans(inputDict, roundN, percentum=True):
    """
    控制评估字典精度输出
    :param inputDict: 输入字典
    :param roundN: 小数点保留的位数
    :param percentum: 是否要转成百分比
    :return:
    """

    if percentum:
        percent = '%'
        for key, value in inputDict.items():
            if key == 'TrainWhole' or key == 'DevWhole':
                for subkey, subvalue in inputDict[key].items():
                    inputDict[key][subkey] = str(round(subvalue/100, roundN)) + percent
            if key == 'TrainSubclass' or key == 'DevSubclass':
                for subkey, subvalue in inputDict[key].items():
                    for sub2key, sub2value in inputDict[key][subkey].items():
                        inputDict[key][subkey][sub2key] = str(round(sub2value/100, roundN)) + percent
    else:
        for key, value in inputDict.items():
            if key == 'TrainWhole' or key == 'DevWhole':
                for subkey, subvalue in inputDict[key].items():
                    inputDict[key][subkey] = round(subvalue/100, roundN)
            if key == 'TrainSubclass' or key == 'DevSubclass':
                for subkey, subvalue in inputDict[key].items():
                    for sub2key, sub2value in inputDict[key][subkey].items():
                        inputDict[key][subkey][sub2key] = round(sub2value/100, roundN)

    return inputDict


def printConfig(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def makePath(params):
    """
    Make folders for training and evaluation,
    """
    if not os.path.isdir(params.resultPath):
        os.makedirs(params.resultPath)
    if not os.path.isdir(params.ckptPath):
        os.makedirs(params.ckptPath)
    if not os.path.isdir("log"):
        os.makedirs("log")


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocabFile):
        os.remove(params.vocabFile)

    if os.path.isfile(params.mapFile):
        os.remove(params.mapFile)

    if os.path.isdir(params.ckptPath):
        shutil.rmtree(params.ckptPath)

    if os.path.isdir(params.summaryPath):
        shutil.rmtree(params.summaryPath)

    if os.path.isdir(params.resultPath):
        shutil.rmtree(params.resultPath)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.configFile):
        os.remove(params.configFile)

    if os.path.isfile(params.vocabFile):
        os.remove(params.vocabFile)


def saveConfig(config, configFile):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(configFile, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def loadConfig(configFile):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(configFile, encoding="utf8") as f:
        return json.load(f)


def convertToText(line):
    """
    Convert conll data to text
    """
    toPrint = []
    for item in line:

        try:
            if item[0] == " ":
                toPrint.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                toPrint.append("[")
            toPrint.append(word)
            if tag[0] in "SE":
                toPrint.append("@" + tag.split("-")[-1])
                toPrint.append("]")
        except:
            print(list(item))
    return "".join(toPrint)


def saveModel(sess, model, path, logger):
    checkpointPath = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpointPath)
    logger.info("model saved")


def createModel(session, modelClass, path, loadVec, config, idToChar, logger, dependencePath, isTrain):
    # create model, reuse parameters if exists
    model = modelClass(config)
    if isTrain:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["preEmb"]:
            embWeights = session.run(model.charLookup.read_value())
            basename = os.path.basename(config["embFile"])
            embWeights = loadVec(os.path.join(dependencePath, basename), idToChar, config["charDim"], embWeights)
            session.run(model.charLookup.assign(embWeights))
            logger.info("Load pre-trained embedding.")
        else:
            raise Exception("{} is not exists!Create model failed during train!".format(config["preEmb"]))

    if not isTrain:#预测
        print(path)
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            try:
                model.saver.restore(session, ckpt.model_checkpoint_path)
            except Exception() as e:
                raise e
        else:
            raise Exception("{} is not exists!".format(ckpt.model_checkpoint_path))
    return model

def transTags(entity):
    """将一个完整实体tag转成标准BIOES格式"""
    if len(entity) == 1:
        return ["S" + entity[0][1:]]
    elif len(entity) == 2:
        return ["B" + entity[0][1:], "E" + entity[1][1:]]
    elif len(entity) > 2:
        entity[0] = "B" + entity[0][1:]
        entity[-1] = "E" + entity[-1][1:]
        return entity
    else:
        return entity

def resultToJson(string, tags, ids):
    # import logging
    # logger = logging.getLogger("django")

    # 检查并修正tags逻辑
    trueTags = []
    fullEntity = []
    for index, tag in enumerate(tags):
        char = string[index]
        if len(fullEntity) == 0:
            if tag[0] in ["S", "B", "I", "E"]:
                fullEntity.append(tag)
            else:
                trueTags.append(tag)
        else:
            if tag[0] in ["S", "B", "I", "E"]:
                if tag[0] == "S":
                    fullEntity = transTags(fullEntity)
                    trueTags += fullEntity
                    fullEntity = []
                    trueTags.append(tag)
                elif tag[0] == "E":
                    fullEntity.append(tag)
                    fullEntity = transTags(fullEntity)
                    trueTags += fullEntity
                    fullEntity = []
                elif tag[0] == "B":
                    fullEntity = transTags(fullEntity)
                    trueTags += fullEntity
                    fullEntity = [tag]
                else:
                    fullEntity.append(tag)
            else:
                fullEntity = transTags(fullEntity)
                trueTags += fullEntity
                fullEntity = []
                trueTags.append(tag)
    if len(fullEntity) != 0:
        fullEntity = transTags(fullEntity)
        trueTags += fullEntity
        fullEntity = []
    assert len(trueTags) == len(tags), "resultToJson函数tags校正错误!"
    tags = trueTags
    
    item = {"string": string, "entities": []}
    # for (key,idValue) in ids.items():
    #     item[key] = idValue
    # print('item:',item)
    #entList = []
    entityName = ""
    entityStart = 0
    idx = 0
    strList = []
    bioList=[]
    for char, tag in zip(string, tags):
        bioList.append(tag)
        strList.append(char)
        if tag[0] == "S":
            #entList.append({"string": string,"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            item["entities"].append({"string": string,"word": char, "start": idx, "end": idx+1, "type":tag[2:],ids[0]:ids[1]})
        elif tag[0] == "B":
            entityName += char
            entityStart = idx
        elif tag[0] == "I":
            entityName += char
        elif tag[0] == "E":
            entityName += char
            item["entities"].append({"string": string,"word": entityName, "start": entityStart, "end": idx + 1, "type": tag[2:],ids[0]:ids[1]})
            entityName = ""
            #entList.append({"string": string,"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        else:
            entityName = ""
            entityStart = idx
        idx += 1
    #item["entities"] = entList
    item["bio"] = {"bio":bioList,"char":strList,"string":string}
    item["bio"] = {"bio":bioList,"char":strList}
    return item






