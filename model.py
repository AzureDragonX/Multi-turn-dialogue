import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
import rnncell as rnn
import utils as utils
import dataUtils as dataUtils

import os
import sys
import importlib
BaseDir = os.path.dirname(os.path.abspath(__file__))
if "afw_ai_engine" in BaseDir:
    BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai_engine.",1)[-1]
else:
    BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai.",1)[-1]
# rnn = importlib.import_module(BaseImportDir+".rnncell")
# utils = importlib.import_module(BaseImportDir+".utils")
resultToJson = utils.resultToJson
# dataUtils = importlib.import_module(BaseImportDir+".dataUtils")
createInput = dataUtils.createInput
iobesIob = dataUtils.iobesIob


class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config["learningRate"]
        self.charDim = config["charDim"]
        self.lstmDim = config["lstmDim"]
        self.segDim = config["segDim"]

        self.numTags = config["numTags"]
        self.numChars = config["numChars"]
        self.numSegs = 4

        self.globalStep = tf.Variable(0, trainable=False)
        self.bestDevF1 = tf.Variable(0.0, trainable=False)
        self.bestTestF1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.charInputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.segInputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.charInputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batchSize = tf.shape(self.charInputs)[0]
        self.numSteps = tf.shape(self.charInputs)[-1]

        # embeddings for chinese character and segmentation representation
        embedding = self.embeddingLayer(self.charInputs, self.segInputs, config)

        # apply dropout before feed to lstm layer
        lstmInputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstmOutputs = self.biLSTMLayer(lstmInputs, self.lstmDim, self.lengths)

        # logits for tags
        self.logits = self.projectLayer(lstmOutputs)

        # loss of the model
        self.loss = self.lossLayer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "SGD":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "ADAM":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "ADAGRAD":#adagrad
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            gradsVars = self.opt.compute_gradients(self.loss)
            cappedGradVars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in gradsVars]
            self.trainOp = self.opt.apply_gradients(cappedGradVars, self.globalStep)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embeddingLayer(self, charInputs, segInputs, config, name=None):
        """
        :param charInputs: one-hot encoding of sentence
        :param segInputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, numSteps, embedding size], 
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name):
            self.charLookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.numChars, self.charDim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.charLookup, charInputs))
            if config["segDim"]:
                with tf.variable_scope("seg_embedding"):
                    self.segLookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.numSegs, self.segDim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.segLookup, segInputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTMLayer(self, lstmInputs, lstmDim, lengths, name=None):
        """
        :param lstmInputs: [batchSize, numSteps, emb_size] 
        :return: [batchSize, numSteps, 2*lstmDim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstmCell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstmCell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstmDim,
                        usePeepholes=True,
                        initializer=self.initializer,
                        stateIsTuple=True)
            outputs, finalStates = tf.nn.bidirectional_dynamic_rnn(
                lstmCell["forward"],
                lstmCell["backward"],
                lstmInputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def projectLayer(self, lstmOutputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstmOutputs: [batchSize, numSteps, emb_size] 
        :return: [batchSize, numSteps, numTags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstmDim*2, self.lstmDim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstmDim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstmOutputs, shape=[-1, self.lstmDim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstmDim, self.numTags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.numTags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.numSteps, self.numTags])

    def lossLayer(self, projectLogits, lengths, name=None):
        """
        calculate crf loss
        :param projectLogits: [1, numSteps, numTags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            startLogits = tf.concat(
                [small * tf.ones(shape=[self.batchSize, 1, self.numTags]), tf.zeros(shape=[self.batchSize, 1, 1])], axis=-1)
            padLogits = tf.cast(small * tf.ones([self.batchSize, self.numSteps, 1]), tf.float32)
            logits = tf.concat([projectLogits, padLogits], axis=-1)
            logits = tf.concat([startLogits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.numTags*tf.ones([self.batchSize, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.numTags + 1, self.numTags + 1],
                initializer=self.initializer)
            logLikelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-logLikelihood)

    def createFeedDict(self, isTrain, batch):
        """
        :param isTrain: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feedDict = {
            self.charInputs: np.asarray(chars),
            self.segInputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if isTrain:
            feedDict[self.targets] = np.asarray(tags)
            feedDict[self.dropout] = self.config["dropoutKeep"]
        return feedDict

    def runStep(self, sess, isTrain, batch):
        """
        :param sess: session to run the batch
        :param isTrain: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feedDict = self.createFeedDict(isTrain, batch)
        if isTrain:
            globalStep, loss, _ = sess.run(
                [self.globalStep, self.loss, self.trainOp],
                feedDict)
            return globalStep, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feedDict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batchSize, numSteps, numTags]float32, logits
        :param lengths: [batchSize]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.numTags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, dataManager, idToTag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param idToTag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in dataManager.iterBatch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.runStep(sess, False, batch)
            batchPaths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobesIob([idToTag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobesIob([idToTag[int(x)] for x in batchPaths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluateLine(self, sess, inputs, idToTag,ids):
        trans = self.trans.eval()
        lengths, scores = self.runStep(sess, False, inputs)
        batchPaths = self.decode(scores, lengths, trans)
        tags = [idToTag[idx] for idx in batchPaths[0]]
        return resultToJson(inputs[0][0], tags,ids)
