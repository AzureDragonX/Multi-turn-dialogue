# encoding = utf8
import re
import math
import codecs
import random

import numpy as np
import jieba
jieba.initialize()


def createDico(itemList):
    """
    Create a dictionary of items from a list of list of items.
    创建一个字典，key是字符，值是字符出现的次数。
    """
    assert type(itemList) is list
    dico = {}
    for items in itemList:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def createMapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sortedItems = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    idtoItem = {i: v[0] for i, v in enumerate(sortedItems)}
    itemToId = {v: k for k, v in idtoItem.items()}
    return itemToId , idtoItem


def zeroDigits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iobIobes(tags):
    """
    IOB -> IOBES
    """
    newTags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            newTags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                newTags.append(tag)
            else:
                newTags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                newTags.append(tag)
            else:
                newTags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return newTags


def iobesIob(tags):
    """
    IOBES -> IOB
    """
    newTags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            newTags.append(tag)
        elif tag.split('-')[0] == 'I':
            newTags.append(tag)
        elif tag.split('-')[0] == 'S':
            newTags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            newTags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            newTags.append(tag)
        else:
            raise Exception('Invalid format!')
    return newTags


def insertSingletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    newWords = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            newWords.append(0)
        else:
            newWords.append(word)
    return newWords


def getSegFeatures(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    segFeature = []
    for word in jieba.cut(string):
        # print(word)
        if len(word) == 1:
            segFeature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            segFeature.extend(tmp)
    # print(segFeature)
    return segFeature


def createInput(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def loadWord2vec(embPath, idToWord, wordDim, oldWeights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    newWeights = oldWeights
    print('Loading pretrained embeddings from {}...'.format(embPath))
    preTrained = {}
    embInvalid = 0
    for i, line in enumerate(codecs.open(embPath, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == wordDim + 1:
            preTrained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            embInvalid += 1
    if embInvalid > 0:
        print('WARNING: %i invalid lines' % embInvalid)
    cFound = 0
    cLower = 0
    cZeros = 0
    nWords = len(idToWord)
    # Lookup table initialization
    for i in range(nWords):
        word = idToWord[i]
        if word in preTrained:
            newWeights[i] = preTrained[word]
            cFound += 1
        elif word.lower() in preTrained:
            newWeights[i] = preTrained[word.lower()]
            cLower += 1
        elif re.sub('\d', '0', word.lower()) in preTrained:
            newWeights[i] = preTrained[
                re.sub('\d', '0', word.lower())
            ]
            cZeros += 1
    print('Loaded %i pretrained embeddings.' % len(preTrained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        cFound + cLower + cZeros, nWords,
        100. * (cFound + cLower + cZeros) / nWords)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        cFound, cLower, cZeros
    ))
    return newWeights


def fullToHalf(s):
    """
    Convert full-width character to half-width one 
    """

    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cutToSentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    lenP = len(text)
    preCut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if preCut:
            cut=True
            preCut=False
        if word in u"。;!?\n":
            cut = True
            if lenP > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    preCut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replaceHtml(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def inputFromLine(line, charToId):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = fullToHalf(line)
    line = replaceHtml(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[charToId[char] if char in charToId else charToId["<UNK>"]
                   for char in line]])
    inputs.append([getSegFeatures(line)])
    inputs.append([[]])
    return inputs


class BatchManager(object):

    def __init__(self, data, batchSize):
        self.batchData = self.sortAndPad(data, batchSize)
        self.lenData = len(self.batchData)

    def sortAndPad(self, data, batchSize):
        numBatch = int(math.ceil(len(data) /batchSize))
        sortedData = sorted(data, key=lambda x: len(x[0]))
        batchData = list()
        for i in range(numBatch):
            batchData.append(self.padData(sortedData[i*int(batchSize) : (i+1)*int(batchSize)]))
        return batchData

    @staticmethod
    def padData(data):
        strings = []
        chars = []
        segs = []
        targets = []
        maxLength = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (maxLength - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iterBatch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batchData)
        for idx in range(self.lenData):
            yield self.batchData[idx]
