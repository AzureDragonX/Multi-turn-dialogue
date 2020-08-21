import re
import codecs
import numpy as np

import os
import sys
import importlib
import dataUtils as dataUtils
BaseDir = os.path.dirname(os.path.abspath(__file__))
# if "afw_ai_engine" in BaseDir:
#     BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai_engine.",1)[-1]
# else:
#     BaseImportDir = BaseDir.replace("\\",'.').replace("/",'.').rsplit("afw_ai.",1)[-1]
# dataUtils = importlib.import_module(BaseImportDir+".dataUtils")
createDico = dataUtils.createDico
createMapping = dataUtils.createMapping
zeroDigits = dataUtils.zeroDigits
iob2 = dataUtils.iob2
iobIobes = dataUtils.iobIobes
getSegFeatures = dataUtils.getSegFeatures



def loadSentences(df, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for i, line in df.iterrows():
        if (line[0] is np.nan or line[0] == '') and (line[1] == '' or line[1] is np.nan):
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] is None or line[1] is None or line[0] is np.nan or line[1] is np.nan or line[0] == '' or line[1] == '':
                continue
            first = zeroDigits(line[0].rstrip()) if zeros else line[0].rstrip()
            second = zeroDigits(line[1].rstrip()) if zeros else line[1].rstrip()
            sentence.append([first, second])
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def updateTagScheme(sentences, tagScheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    # print(sentences)
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            sStr = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, sStr))
        if tagScheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, newTag in zip(s, tags):
                word[-1] = newTag
        elif tagScheme == 'iobes':
            newTags = iobIobes(tags)
            for word, newTag in zip(s, newTags):
                word[-1] = newTag
        else:
            raise Exception('Unknown tagging scheme!')


def charMapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    创建一个字典索引映射关系，在charToId字典中，key是字符，value是字符出现次数逆序排列的索引位置。
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences] #取出每个字符[['患', '者', '于', '4', '月', '余',...]]
    dico = createDico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    charToId, idToChar = createMapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, charToId, idToChar


def tagMapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    构建标签字典
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = createDico(tags)
    tagToId, idToTag = createMapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tagToId, idToTag


def prepareDataset(sentences, charToId, tagToId, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
        seg:是jieba的特征
        tag:标志的id
    """

    noneIndex = tagToId["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for sentence in sentences:
        senTmp = []
        for sen in sentence:
            if "" != sen[0]:#去掉sentence中的空字符
                senTmp.append(sen)
        if senTmp == []:# 如果整个句子为空,则直接去掉,不然会影响结果
            continue
        sentence = senTmp
        string = [w[0] for w in sentence]
        chars = [charToId[f(w) if f(w) in charToId else '<UNK>']
                 for w in string]
        segs = getSegFeatures("".join(string))#如果sentence中存在""空字符,数组长度会不一致
        if train:
            tags = [tagToId[w[-1]] for w in sentence]
        else:
            tags = [noneIndex for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augmentWithPretrained(dictionary, extEmbPath, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % extEmbPath)
    assert os.path.isfile(extEmbPath)
    # Load pretrained embeddings from file
    # pretrained预训练的字符嵌入字符集
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(extEmbPath, 'r', 'utf-8')
        if len(extEmbPath) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    wordToId, idToWord = createMapping(dictionary)
    return dictionary, wordToId, idToWord


def saveMaps(savePath, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(savePath, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def loadMaps(savePath):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(savePath, "r", encoding="utf8") as f:
    #     pickle.load(savePath, f)

