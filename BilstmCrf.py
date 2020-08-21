# -*- coding:utf-8 -*-
import sys
import os
import importlib
import pickle
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
from collections import OrderedDict
import itertools
import numpy as np
import os
import json
import math
import re
import loader as loader
import dataProcess as dataProcess
import dataUtils as dataUtils
import utils as utils
import model as model

import traceback
import afw.logging as trainLogging
from afw.OutputStandard import normalTable
from afw.StandardizedData import stdOutputDF,stdSourceData,stdNormalTable
import itertools
import time
# import logging
# logger = logging.getLogger("django")
# logger.info("当前的搜索路径为：{}".format(sys.path))
import functools
import time
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - begin_time
        print('{} 共用时：{} s'.format(func.__name__, run_time))
        return value
    return wrapper


class importDependence():
    def imports(self):
        startTime = time.time()
        self.BaseDir = os.path.dirname(os.path.abspath(__file__))
        # if "afw_ai_engine" in self.BaseDir:
        #     self.BaseImportDir = self.BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai_engine.",1)[-1]
        # else:
        #     self.BaseImportDir = self.BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai.",1)[-1]
        self.BaseImportDir = self.BaseDir
        print('BaseDir:', self.BaseDir)
        print('BaseImportDir:', self.BaseImportDir)

        #loader
        # loader = importlib.import_module(self.BaseImportDir+"\\loader")
        self.loadSentences = loader.loadSentences
        self.updateTagScheme = loader.updateTagScheme
        self.charMapping = loader.charMapping
        self.tagMapping = loader.tagMapping
        self.augmentWithPretrained = loader.augmentWithPretrained
        self.prepareDataset = loader.prepareDataset
        
        #dataProcess
        # dataProcess = importlib.import_module(self.BaseImportDir+".dataProcess")
        self.dataProcess = dataProcess

        #dataUtils
        # dataUtils = importlib.import_module(self.BaseImportDir+".dataUtils")
        self.loadWord2vec = dataUtils.loadWord2vec
        self.createInput = dataUtils.createInput
        self.inputFromLine = dataUtils.inputFromLine
        self.BatchManager = dataUtils.BatchManager
        
        #utils
        # utils = importlib.import_module(self.BaseImportDir+".utils")
        self.printConfig = utils.printConfig
        self.saveConfig = utils.saveConfig
        self.loadConfig = utils.loadConfig
        self.testNer = utils.testNer
        self.stdoutTrans = utils.stdoutTrans
        self.makePath = utils.makePath
        self.clean = utils.clean
        self.createModel = utils.createModel
        self.saveModel = utils.saveModel
        self.utils = utils

        #model
        # model = importlib.import_module(self.BaseImportDir+".model")
        self.Model = model.Model
        print('耗时：',time.time()-startTime)
        return self

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

class PredictModel:
    def __init__(self, **kwargs):
        import logging
        self.logger = logging.getLogger("django")
        self.curDep = importDependence()
        self.curDep.imports()

        modelPath = kwargs['modelPath']#注意这里的modelPath是模型bin文件的地址，与train的modelPath可能不同
        dependencetPath = os.path.join(self.curDep.BaseDir,"dependence")
        with open(os.path.join(modelPath, "maps.pkl"), "rb") as file:
            self.charToId, self.idToChar, self.tagToId, self.idToTag = pickle.load(file)
        tf.reset_default_graph()#清除默认图形堆栈并重置全局默认图形
        self.graph = tf.Graph()
        cudaValue = str(kwargs["cudaValue"]) if "cudaValue" in kwargs.keys() else "-1"
        cpuGpu = "/cpu:0" if cudaValue == "-1" else "/gpu:{}".format(cudaValue)
        with self.graph.as_default():
            with tf.device('{}'.format(cpuGpu)):
                self.logger.info("当前使用的Graph是{}".format(self.graph))
                config = self.curDep.loadConfig(os.path.join(modelPath, "configFile"))
                tfConfig = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
                tfConfig.gpu_options.allow_growth = True
                tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.3
                self.session = tf.Session(graph=self.graph, config=tfConfig)
                self.model = self.curDep.createModel(self.session, self.curDep.Model, modelPath, self.curDep.loadWord2vec, config, self.idToChar, None, dependencetPath, False)

    def TEST(self,ColumnsList, chartType, title, contentList):
        if chartType == "normal_table":
            if contentList!=[]:
                if len(ColumnsList) != contentList.shape[1]:
                    assert 'ColumnsList与数据不匹配'
                else:
                    content = []
                    for i in range(len(contentList)):
                        temp_dict = {}
                        for j in range(len(ColumnsList)):
                            temp_dict[ColumnsList[j]] = contentList[i, j]
                        content.append(temp_dict)

                    output = {"chartType": chartType,
                              "title": title,
                              "table": {"columns": ColumnsList,
                                        "content": content
                                        }
                              }
                return output
            else:
                 output = {"chartType": chartType,
                              "title": title,
                              "table": {"columns": "columns为空",
                                        "content": []
                                        }
                              }
                 return output

    @timer
    def predict(self, **kwargs):
        '''
        @description: 命名实体识别BilstmCrf算法-预测
        @param {type} 
        @return: chartypeResult dict对象 对象。对前端的输出格式，可渲染为符合前端规范的图表。同时存为数据集。;
                 result1 不限 普通输出，不限制类型。组合接口组合时调用，所有实体识别算法的这个输出都必须统一格式。;
                 result2 dataFrame 具备"string"、"word"、“start"、”end“、”type“和”id“六列的dataframe对象。. 
        '''
        try:
            self.logger.info("kwargs: {}".format(kwargs.keys()))
            inputData = kwargs.get("inputData", None)
            inputColumns = kwargs.get("inputColumns", None)
            idInfo = kwargs.get("idInfo", None)

            if inputData is not None and isinstance(inputData, dict): # inputData
                if inputColumns is None or inputColumns[0] not in inputData.keys():
                    # 默认inputData中包含原文数据列data。
                    assert 'data' in inputData.keys(), "inputData中找不到inputColumns输入数据列！"
                    inputColumns = ['data']
                else:
                    inputColumns = [inputColumns[0]]
                if idInfo is None or idInfo[0] not in inputData.keys():
                    idInfo = ['id']
                else:
                    idInfo = [idInfo[0]]

                dataLength = len(inputData[inputColumns[0]])
                if idInfo[0] not in inputData.keys():
                    inputData[idInfo[0]] = list(range(dataLength))
                inputKeys = inputData.keys()
                for key in inputKeys:
                    assert  len(inputData[key]) == dataLength, "inputData len({}) != len(data).".format(key)
                sourceData = pd.DataFrame().from_dict(inputData)
                idToSession = None
                if "session_id" in inputKeys:
                    idToSession = dict()
                    for id,session in zip(inputData[idInfo[0]], inputData["session_id"]):
                        idToSession[str(id)] = session
            else:# testData
                testData = kwargs["testData"]
                testData.columns = [column.strip() for column in testData.columns]
                testData = stdSourceData(testData)
                assert inputColumns is not None, "参数inputColumns为空！"
                inputColumns = [column.strip() for column in inputColumns]

                if idInfo is None or idInfo == []:
                    if 'index' not in testData.columns:
                        testData['index'] = list(range(len(testData)))
                    idInfo = ['index']
                sourceData = testData[inputColumns+idInfo]
        except Exception as e:
            traceback.print_exc(self.logger.debug(e))
            raise Exception("BilstmCrf获取数据：{}".format(e))

        self.logger.info("sourceData length: {}".format(len(sourceData)))

        specialSymbol = "[\r\n|\n]" #去除句子中的换行符
        results = []
        bioResults = []
        with self.session.as_default():
            for index, line in sourceData.iterrows():
                if index % 1000 == 0:
                    self.logger.info("Index NO.: {}".format(index))
                string = line[inputColumns[0]]
                if string is "" or string is np.nan:
                    continue
                ids = [idInfo[0], line[idInfo[0]]] # [id, xxxx]
                string = re.sub(specialSymbol, "", string)
                string = string.strip()
                if string == "":
                    continue
                result = self.model.evaluateLine(self.session, self.curDep.inputFromLine(string, self.charToId),
                                                 self.idToTag, ids)
                results.append(result['entities'])
                bioResults.append(result['bio'])
        contentList = np.array(list(itertools.chain.from_iterable([[[v for i,v in line.items()] for line in sample ]for sample in results])))

        # 如果返回实体为空，没有实体。
        chartType = "normal_table"
        title = "命名实体识别"
        ColumnsList = ["string", "word", "start", "end", "type", "id"]
        # 特殊处理, 一个实体也没有，则返回空字符串
        if contentList.size == 0:
            contentList = np.array([["", "", "", "", "", ""]])
            outputType0 = self.TEST(ColumnsList, chartType, title, contentList)
            outputType2 = pd.DataFrame(contentList, columns=ColumnsList)
        else:
            outputType0 = self.TEST(ColumnsList, chartType, title, contentList)
            outputType2 = pd.DataFrame(contentList, columns=ColumnsList)
            outputType0 = stdNormalTable(outputType0)
            outputType2 = stdOutputDF(outputType2)
        columns = outputType0['table']["columns"]
        content = outputType0['table']["content"]
        outputType1Data = []
        for line in content:
            outputType1Data.append(line.values())
        outputType1 = pd.DataFrame(outputType1Data, columns=columns)
        outputType3 = pd.DataFrame(outputType0["table"]["content"], columns=outputType0["table"]["columns"])

        # 如果是单条输入做特殊处理
        if inputData is not None and idToSession is not None:
            temp = pd.DataFrame()
            if len(outputType2) == 1 and outputType2.loc[0, "word"] == "":#单输入没有识别出实体
                for index, line in outputType2.iterrows():
                    line["string"] = inputData['data'][0]
                    line["session_id"] = inputData['session_id'][0]
                    temp = temp.append(line)
                outputType2 = temp
            else:
                for index, line in outputType2.iterrows():
                    line["session_id"] = idToSession[str(line["id"])]
                    temp = temp.append(line)
                outputType2 = temp
        outputType2
        return outputType0, outputType1, outputType2, outputType3

class TrainModel:

    def __init__(self, **kwargs):

        self.curDep = importDependence()
        self.curDep.imports()
        self.current_epoch = 0
        tf.reset_default_graph()
        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS
        # 同一进程多次实例化train，定义的attrs没有被清除，会出现重复定义问题，每次训练前先做一次清除操作。
        attrs = list(self.FLAGS._flags().keys())
        for attr in attrs:
            self.FLAGS.__delattr__(attr)

        self.tfConfig = tf.ConfigProto(allow_soft_placement = True)
        self.tfConfig.gpu_options.allow_growth = True
        self.tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.6

        logFile = kwargs['logPath']
        self.flags.DEFINE_string("logFile", logFile, "File for log")
        self.flags.DEFINE_boolean("clean", False, "clean train folder")
        if self.FLAGS.clean:
            self.curDep.clean(self.FLAGS)
            os.makedirs(logFile)
            logF = open(self.FLAGS.logFile, "w")
            logF.close()
            self.logger = trainLogging.Logger(self.FLAGS.logFile)
            self.logger.info("Train Starting, clean all files.")
        else:
            self.logger = trainLogging.Logger(self.FLAGS.logFile)

        self.graph = tf.get_default_graph()
        self.session = tf.get_default_session()

    # config for the model
    def __configModel(self, charToId, tagToId):
        config = OrderedDict()
        config["numChars"] = len(charToId)
        config["charDim"] = self.FLAGS.charDim
        config["numTags"] = len(tagToId)
        config["segDim"] = self.FLAGS.segDim
        config["lstmDim"] = self.FLAGS.lstmDim
        config["batchSize"] = self.FLAGS.batchSize

        config["embFile"] = self.FLAGS.embFile
        config["clip"] = self.FLAGS.clip
        config["dropoutKeep"] = 1.0 - self.FLAGS.dropout
        config["optimizer"] = self.FLAGS.optimizer
        config["learningRate"] = self.FLAGS.learningRate
        config["tagSchema"] = self.FLAGS.tagSchema
        config["preEmb"] = self.FLAGS.preEmb
        config["zeros"] = self.FLAGS.zeros
        config["lower"] = self.FLAGS.lower
        config["epoch"] = self.FLAGS.epoch
        return config


    def __evaluate(self, sess, model, name, data, idToTag, logger):
        evaluateResult = dict()
        logger.info("evaluate:{}".format(name))
        nerResults = model.evaluate(sess, data, idToTag) # 预测结果，格式【原始字符 正确标注 预测标注结果】
        evalResult, evalResultDict = self.curDep.testNer(nerResults, self.FLAGS.resultPath)
        for line in evalResult:#打印出来
            logger.info(line)

        wholeDict = evalResultDict['wholeClass']
        subClassDict = evalResultDict["subClass"]

        if name == 'dev':
            evaluateResult['TrainWhole'] = wholeDict
            evaluateResult['TrainSubclass'] = subClassDict
        if name == 'test':
            evaluateResult['DevWhole'] = wholeDict
            evaluateResult['DevSubclass'] = subClassDict

        f1 = evalResultDict["wholeClass"]['MacroF1']
        if name == "dev":
            bestDevF1 = model.bestDevF1.eval()
            if f1 > bestDevF1:
                tf.assign(model.bestDevF1, f1).eval()  # tf.assign()更新赋值,把best更新为f1.
                logger.info("new best dev f1 score:{:>.3f}".format(f1))
            return f1 > bestDevF1, evaluateResult
        elif name == "test":
            bestTestF1 = model.bestTestF1.eval()
            if f1 > bestTestF1:
                tf.assign(model.bestTestF1, f1).eval()
                logger.info("new best test f1 score:{:>.3f}".format(f1))

            return f1 > bestTestF1, evaluateResult

    def bratToBio(self, dfData):
        import logging
        logger = logging.getLogger("django")
        try:
            logger.info("Enter bratToBio!")
            columns = ["id", "sourcedata", "word", "type", "startPosition", "endPosition"]
            dfData = dfData[columns]
            # 传进来的可能出现空行包含nan,inf，去除空行数据
            dfData = dfData[pd.notna(dfData['id'])]
            dfData = dfData[pd.notna(dfData['sourcedata'])]
            dfData = dfData[pd.notna(dfData['word'])]
            dfData = dfData[pd.notna(dfData['type'])]
            dfData = dfData[pd.notna(dfData['startPosition'])]
            dfData = dfData[pd.notna(dfData['endPosition'])]
            dfData[["startPosition", "endPosition"]] = dfData[["startPosition", "endPosition"]].astype(int)
            dfData = dfData.sort_values(['id', 'startPosition', 'endPosition'], ascending=[True, True, True])
            ids = sorted(set(dfData['id']))
            datas = []
            labels = []
            errFlag = False
            for sampleId in ids:
                idDf = dfData[dfData['id'] == sampleId]
                idDf = idDf.sort_values(['startPosition', 'endPosition'], ascending=[True, True])
                sources = list(set(idDf['sourcedata']))
                if len(sources) != 1:
                    assert "同个id出现多个不同原文！！"
                sourcedata = sources[0]
                label = ['O'] * len(sourcedata)
                data = list(sourcedata)
                for index, line in idDf.iterrows():
                    dataId = line['id']
                    dataType = line['type']
                    word = line['word']
                    startPos = int(line['startPosition'])
                    endPos = int(line['endPosition'])
                    if not sourcedata[startPos:endPos+1] == word:
                        logger.info("ID:{} word:{} position check failed!!!".format(dataId, word))
                        errFlag = True
                        break
                    for i in range(startPos, endPos+1, 1):
                        if i == startPos:
                            label[i] = "B-" + dataType
                        else:
                            label[i] = "I-" + dataType
                if errFlag:#出现配置信息错误
                    logger.info("Sample ID:{} sourcedata:{} position errors!!!".format(sampleId, sourcedata))
                    errFlag = False
                    continue
                for index, line_data in enumerate(data):
                    if line_data in ['。', '?', '？', '!', '！']:
                        data[index] = ''
                        label[index] = ''
                # 换样本
                data.append("")
                label.append("")
                if len(data) == len(label):
                    datas += data
                    labels += label
            df = pd.DataFrame([x for x in zip(datas, labels)], columns=["data", "label"])
            logger.info("BratToBio Finished!")
        except Exception as e:
            logger.info(e)
        return df

    @timer
    def train(self, **kwargs):

        try:
            import logging
            logger = logging.getLogger("django")
            logger.info("==========================BilstmCrf训练==================================")
            trainData = kwargs['trainData']
            trainData.columns = [column.strip() for column in trainData.columns]
            devData = kwargs['devData']
            devData.columns = [column.strip() for column in devData.columns]
            cols0 = ["id", "sourcedata", "word", "type", "startPosition", "endPosition"]
            dataCols = kwargs['dataColumns']["data"]
            labelCol = kwargs['dataColumns']["label"]
            cols = dataCols + labelCol
            if set(cols0).issubset(set(trainData.columns)) and set(cols0).issubset(set(devData.columns)):
                trainData = self.bratToBio(trainData)
                devData = self.bratToBio(devData)
            elif set(cols).issubset(set(trainData.columns)) and set(cols).issubset(set(devData.columns)):
                trainData = trainData[cols]
                devData = devData[cols]
            else:
                self.logger.info("the columns of datasets is invalid!")
            modelPath = kwargs['modelPath']
            tensorboardPath = kwargs['tensorboardPath']
            batchSize = kwargs['batchSize']
            learningRate = kwargs['learningRate']
            optimizer = kwargs['optimizer']
            epoch = kwargs['epoch']
            dependencetPath = os.path.join(self.curDep.BaseDir,"dependence")
        except Exception as e:
            #print(e.args)
            traceback.print_exc(file=open(self.FLAGS.logFile, "a+"))

        self.flags.DEFINE_boolean("train", True, "Wither train the model")
        self.flags.DEFINE_string("tagSchema", "iobes", "tagging schema iobes or iob")
        self.flags.DEFINE_string("mapFile", os.path.join(modelPath, "maps.pkl"), "file for maps")
        self.flags.DEFINE_boolean("lower", True, "Wither lower case")

        self.logger.info(self.curDep.BaseDir)

        self.flags.DEFINE_string("embFile", os.path.join(dependencetPath,"wiki_100.utf8"), "Path for pre_trained embedding")
        self.flags.DEFINE_float("batchSize", batchSize, "batch size")
        self.flags.DEFINE_string("configFile", os.path.join(modelPath, "configFile"), "File for config")
        self.flags.DEFINE_string("ckptPath", modelPath, "Path to save model")
        self.flags.DEFINE_integer("stepsCheck", 100, "steps per checkpoint")
        self.flags.DEFINE_string("resultPath", os.path.join(modelPath, "result"), "Path for results")
        self.flags.DEFINE_string("script", os.path.join(modelPath, "conlleval"), "evaluation script")
        self.flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
        self.flags.DEFINE_float("learningRate", learningRate, "Initial learning rate")
        self.flags.DEFINE_string("optimizer", optimizer, "Optimizer for training")
        self.flags.DEFINE_integer("epoch", epoch, "maximum training epochs")

        # configurations for the model
        self.flags.DEFINE_integer("segDim", 20, "Embedding size for segmentation, 0 if not used")
        self.flags.DEFINE_integer("charDim", 100, "Embedding size for characters")
        self.flags.DEFINE_integer("lstmDim", 100, "Num of hidden units in LSTM")

        # configurations for training
        self.flags.DEFINE_float("clip", 5, "Gradient clip")
        self.flags.DEFINE_float("dropout", 0.5, "Dropout rate")

        self.flags.DEFINE_boolean("preEmb", True, "Wither use pre-trained embedding")

        self.flags.DEFINE_string("summaryPath", tensorboardPath, "Path to store summaries")
        self.flags.DEFINE_string("vocabFile", "vocab.json", "File for vocab")

        assert self.FLAGS.clip < 5.1, "gradient clip should't be too much"
        assert 0 <= self.FLAGS.dropout < 1, "dropout rate between 0 and 1"
        assert self.FLAGS.learningRate > 0, "learning rate must larger than zero"
        assert self.FLAGS.optimizer in ["ADAM", "SGD", "ADAGRAD"]

        if os.path.isfile(self.FLAGS.configFile):
            os.remove(self.FLAGS.configFile) #训练更新配置文件
        if os.path.isfile(self.FLAGS.mapFile):
            os.remove(self.FLAGS.mapFile) #训练更新映射文件

        finalResult = dict()

        writer = tf.summary.FileWriter(self.FLAGS.summaryPath, tf.get_default_graph())
        trainData = trainData[cols]
        devData = devData[cols]

        allData = pd.concat([trainData, devData])
        allData.reset_index(drop=True, inplace=True)

        #根据分句比例切割训练集，避免从句子中切开标注实体。
        nullIndex = []
        oldline = [None, None]
        for index, line in trainData.iterrows():
            if (line[1] != "" or line[1] is not np.nan) and index > 0 and (oldline[1] is np.nan or oldline[1] == ""):
                nullIndex.append(index-1)
            oldline = line
        assert len(nullIndex) > 10, "the num of train sentences is less than 10!"
        splitIndex = nullIndex[int(len(nullIndex) * 0.9)]

        testDataSplit = trainData.iloc[splitIndex:]
        trainDataSplit = trainData.iloc[:splitIndex]
        devDataSplit = devData

        finalResult["SecondConsuming"] = {"Train": len(trainDataSplit), "Dev":len(devDataSplit)}

        trainSentences = self.curDep.loadSentences(trainDataSplit, self.FLAGS.lower, self.FLAGS.zeros)
        devSentences = self.curDep.loadSentences(devDataSplit, self.FLAGS.lower, self.FLAGS.zeros)
        testSentences = self.curDep.loadSentences(testDataSplit, self.FLAGS.lower, self.FLAGS.zeros)
        allSentences = self.curDep.loadSentences(allData, self.FLAGS.lower, self.FLAGS.zeros)

        # Use selected tagging scheme (IOB / IOBES)
        # 把tag从IOB转成IOBES, 训练数据会改变
        self.curDep.updateTagScheme(trainSentences, self.FLAGS.tagSchema)
        self.curDep.updateTagScheme(devSentences, self.FLAGS.tagSchema)
        self.curDep.updateTagScheme(testSentences, self.FLAGS.tagSchema)
        self.curDep.updateTagScheme(allSentences, self.FLAGS.tagSchema)
        # create maps if not exist
        try:
            # create dictionary for word
            if self.FLAGS.preEmb:# 使用预训练的字符嵌入， 增加test集的字符
                dicoCharsTrain = self.curDep.charMapping(trainSentences, self.FLAGS.lower)[0] #这里只取了第一个dico
                dicoChars, charToId, idToChar = self.curDep.augmentWithPretrained(
                    dicoCharsTrain.copy(),
                    self.FLAGS.embFile,
                    list(itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in testSentences])
                    )
                )
            else:
                _c, charToId, idToChar = self.curDep.charMapping(trainSentences, self.FLAGS.lower)

            # Create a dictionary and a mapping for tags
            _t, tagToId, idToTag = self.curDep.tagMapping(allSentences)
            with open(self.FLAGS.mapFile, "wb") as f: #生成maps.pkl映射文件
                #tagToId {'O': 0, 'B-CF': 1, 'E-CF': 2, 'I-CF': 3, 'S-E': 4, 'B-P': 5, 'E-P': 6, 'I-P': 7, 'B-E': 8, 'E-E': 9, 'I-E': 10, 'S-P': 11, 'S-CF': 12}
                pickle.dump([charToId, idToChar, tagToId, idToTag], f) #
        except Exception as e:
            print(e.args)

            self.logger.info(traceback.format_exc())
            self.logger.info("embFile:{}".format(self.FLAGS.embFile))
            self.logger.info("BilstmCrf出错 CASE2")

        try:
            trainData = self.curDep.prepareDataset(
                trainSentences, charToId, tagToId, self.FLAGS.lower
            )
            devData = self.curDep.prepareDataset(
                devSentences, charToId, tagToId, self.FLAGS.lower
            )
            testData = self.curDep.prepareDataset(
                testSentences, charToId, tagToId, self.FLAGS.lower
            )
            self.logger.info("%i / %i / %i sentences in train / dev / test." % (
                len(trainData), 0, len(testData)))

            trainManager = self.curDep.BatchManager(trainData, self.FLAGS.batchSize)
            devManager = self.curDep.BatchManager(devData, 100)
            testManager = self.curDep.BatchManager(testData, 100)
        except Exception as e:
            raise Exception("创建数据索引出现异常！！！")
            # make path for store log and model if not exist
        try:
            self.curDep.makePath(self.FLAGS)
            if os.path.isfile(self.FLAGS.configFile):
                self.logger.info("Using configFile")
                config = self.curDep.loadConfig(self.FLAGS.configFile)
                config["batchSize"] = self.FLAGS.batchSize
                config["learningRate"] = self.FLAGS.learningRate
                config["epoch"] = self.FLAGS.epoch
                config["optimizer"] = self.FLAGS.optimizer
            else:
                self.logger.info("Create config")
                config = self.__configModel(charToId, tagToId)
            self.curDep.makePath(self.FLAGS)
            self.logger.info("update config:".format(config))
            self.curDep.saveConfig(config, self.FLAGS.configFile)
        except Exception as e:
            raise Exception("保存配置文件出错！！！")

        self.curDep.printConfig(config, self.logger)
        cudaValue = str(kwargs["cudaValue"]) if "cudaValue" in kwargs.keys() else "-1"
        cpuGpu = "/cpu:0" if cudaValue == "-1" else "/gpu:{}".format(cudaValue)
        # limit GPU memory
        try:
            stepsPerEpoch = trainManager.lenData
            graph = tf.Graph()
            with graph.as_default():
                self.logger.info("Tensorflow全局Graph是{}, 当前使用的Graph是{}".format(self.graph, graph))
                with tf.device('{}'.format(cpuGpu)):
                    with tf.Session(graph=graph, config=self.tfConfig).as_default() as sess:
                        model = self.curDep.createModel(sess, self.curDep.Model, self.FLAGS.ckptPath, self.curDep.loadWord2vec, config, idToChar, self.logger, dependencetPath, True)
                        self.logger.info("start training")
                        loss = []
                        bestResult = dict()
                        for i in range(self.FLAGS.epoch):
                            self.current_epoch = i + 1
                            self.logger.info("epoch: {}".format(self.current_epoch))
                            for batch in trainManager.iterBatch(shuffle=True):
                                step, batchLoss = model.runStep(sess, True, batch)
                                self.logger.info("epoch: {}, step:{}".format(self.current_epoch, step))
                                #print("epoch: {}, step:{}".format(self.current_epoch, step))
                                loss.append(batchLoss)
                                if step % self.FLAGS.stepsCheck == 0:
                                    iteration = step // stepsPerEpoch + 1
                                    self.logger.info("iteration:{} step:{}/{}, "
                                                "NER loss:{:>9.6f}".format(
                                        iteration, step % stepsPerEpoch, stepsPerEpoch, np.mean(loss)))
                                    loss = []

                            best, evaluateDevResult = self.__evaluate(sess, model, "dev", devManager, idToTag, self.logger)
                            if i == 0:
                                self.curDep.saveModel(sess, model, self.FLAGS.ckptPath, self.logger)
                                best, evaluateTestResult = self.__evaluate(sess, model, "test", testManager, idToTag, self.logger)
                                bestResult["TrainWhole"] = evaluateDevResult["TrainWhole"]
                                bestResult["TrainSubclass"] = evaluateDevResult["TrainSubclass"]
                                bestResult["DevWhole"] = evaluateTestResult["DevWhole"]
                                bestResult["DevSubclass"] = evaluateTestResult["DevSubclass"]

                            if best:
                                self.curDep.saveModel(sess, model, self.FLAGS.ckptPath, self.logger)
                                best, evaluateTestResult = self.__evaluate(sess, model, "test", testManager, idToTag, self.logger)
                                bestResult["TrainWhole"] = evaluateDevResult["TrainWhole"]
                                bestResult["TrainSubclass"] = evaluateDevResult["TrainSubclass"]
                                bestResult["DevWhole"] = evaluateTestResult["DevWhole"]
                                bestResult["DevSubclass"] = evaluateTestResult["DevSubclass"]
                        bestResult = self.curDep.stdoutTrans(bestResult, 2, False) #保留小数点
                        finalResult["Performance"] = bestResult
                        writer.close()
        except Exception as e:
            writer.close()
            raise Exception("{} 训练过程中出现错误！！！".format(e))

        self.curDep.utils.copyFile(dependencetPath, modelPath)
        return finalResult


def Simple():
    # rootDir = "."
    # outputFilename = 'E://Python//projects//NER//trunk_local//output//CT//test.csv'
    outputNer = 'E://Python//projects//NER//trunk//predict//test-NER.csv'
    # filename = os.path.join(rootDir, "data", "MRI", "test", name + ".csv")
    filename = 'E://Python//projects//NER//trunk//modelFile//testdata//17test.csv'
    testData = pd.read_csv(filename)
    columns = testData.columns
    columns = [column.strip() for column in columns]
    testData.columns = columns
    inputColumns = ["questions"]
    idInfo = ["id"]
    modelPath = 'E://Python//projects//NER//trunk//modelFile//model//QA'
    predictKwargs = {"modelPath": modelPath, "testData": testData, "inputColumns": inputColumns, "idInfo": idInfo}
    outputType0, outputType1, outputType2, outputType3 = PredictModel(**predictKwargs).predict(**predictKwargs)
    # import pdb;pdb.set_trace()
    outputType1.to_csv(outputNer, index=False)

def one_line(descrip):
    # rootDir = "."
    # outputFilename = 'E://Python//projects//NER//trunk_local//output//CT//test.csv'
    # outputNer = 'E://Python//projects//NER//trunk//predict//test-NER.csv'
    # filename = os.path.join(rootDir, "data", "MRI", "test", name + ".csv")
    # filename = 'E://Python//projects//NER//trunk//modelFile//testdata//17test.csv'
    # testData = pd.read_csv(filename)
    # columns = testData.columns
    # columns = [column.strip() for column in columns]
    # testData.columns = columns
    inputColumns = ["questions"]
    idInfo = ["id"]
    modelPath = './/modelFile//model//QA'
    predictKwargs = {"modelPath": modelPath, "inputData": {'questions':[descrip]}, "inputColumns": inputColumns, "idInfo": idInfo}
    outputType0, outputType1, outputType2, outputType3 = PredictModel(**predictKwargs).predict(**predictKwargs)
    return outputType1

if __name__ == "__main__":
    # Simple()
    q = '我有发烧怎么办？'
    print(one_line(q))




