# Python version of the evaluation script from CoNLL'00-
# Originates from: https://github.com/spyysalo/conlleval.py


# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re
import codecs
from collections import defaultdict, namedtuple

anySpace = '<SPACE>'


class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):
    def __init__(self):
        self.correctChunk = 0    # number of correctly identified chunks
        self.correctTags = 0     # number of correct chunk tags
        self.foundCorrect = 0    # number of chunks in corpus
        self.foundGuessed = 0    # number of identified chunks
        self.tokenCounter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.tCorrectChunk = defaultdict(int)
        self.tFoundCorrect = defaultdict(int)
        self.tFoundGuessed = defaultdict(int)


def parseArgs(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=anySpace,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def parseTag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(iterable, options=None):
    if options is None:
        options = parseArgs([])    # use defaults

    counts = EvalCounts()
    numFeatures = None       # number of features per line
    inCorrect = False        # currently processed chunks is correct until now
    lastCorrect = 'O'        # previous chunk tag in corpus
    lastCorrectType = ''    # type of previously identified chunk tag
    lastGuessed = 'O'        # previously identified chunk tag
    lastGuessedType = ''    # type of previous chunk tag in corpus

    for line in iterable:
        line = line.rstrip('\r\n')

        if options.delimiter == anySpace:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if numFeatures is None:
            numFeatures = len(features)
        elif numFeatures != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), numFeatures))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessedType = parseTag(features.pop())
        correct, correctType = parseTag(features.pop())
        firstItem = features.pop(0)

        if firstItem == options.boundary:
            guessed = 'O'

        endCorrect = endOfChunk(lastCorrect, correct,
                                    lastCorrectType, correctType)
        endGuessed = endOfChunk(lastGuessed, guessed,
                                   lastGuessedType, guessedType)
        startCorrect = startOfChunk(lastCorrect, correct,
                                        lastCorrectType, correctType)
        startGuessed = startOfChunk(lastGuessed, guessed,
                                       lastGuessedType, guessedType)

        if inCorrect:
            if (endCorrect  and endGuessed and
                lastGuessedType == lastCorrectType):
                inCorrect = False
                counts.correctChunk += 1
                counts.tCorrectChunk[lastCorrectType] += 1
            elif (endCorrect  != endGuessed or guessedType != correctType):
                inCorrect = False

        if startCorrect and startGuessed and guessedType == correctType:
            inCorrect= True

        if startCorrect:
            counts.foundCorrect += 1
            counts.tFoundCorrect[correctType] += 1
        if startGuessed:
            counts.foundGuessed += 1
            counts.tFoundGuessed[guessedType] += 1
        if firstItem != options.boundary:
            if correct == guessed and guessedType == correctType:
                counts.correctTags += 1
            counts.tokenCounter += 1

        lastGuessed = guessed
        lastCorrect = correct
        lastGuessedType = guessedType
        lastCorrectType = correctType

    if  inCorrect :
        counts.correctChunk += 1
        counts.tCorrectChunk[lastCorrectType] += 1

    return counts


def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]


def calculateMetrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculateMetrics(
        c.correctChunk, c.foundGuessed, c.foundCorrect
    )
    byType = {}
    for t in uniq(list(c.tFoundCorrect) + list(c.tFoundGuessed)):
        byType[t] = calculateMetrics(
            c.tCorrectChunk[t], c.tFoundGuessed[t], c.tFoundCorrect[t]
        )
    return overall, byType


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, byType = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.tokenCounter, c.foundCorrect))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.foundGuessed, c.correctChunk))

    if c.tokenCounter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100.*c.correctTags/c.tokenCounter))
        out.write('precision: %6.2f%%; ' % (100.*overall.prec))
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(byType.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, c.tFoundGuessed[i]))


def reportNotPrint(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, byType = metrics(counts)

    c = counts
    finalReport = []
    line = []
    line.append('processed %d tokens with %d phrases; ' %
              (c.tokenCounter, c.foundCorrect))
    line.append('found: %d phrases; correct: %d.\n' %
              (c.foundGuessed, c.correctChunk))
    finalReport.append("".join(line))

    if c.tokenCounter > 0:
        line = []
        line.append('accuracy: %6.2f%%; ' %
                  (100.*c.correctTags/c.tokenCounter))
        line.append('precision: %6.2f%%; ' % (100.*overall.prec))
        line.append('recall: %6.2f%%; ' % (100.*overall.rec))
        line.append('FB1: %6.2f\n' % (100.*overall.fscore))
        finalReport.append("".join(line))

    for i, m in sorted(byType.items()):
        line = []
        line.append('%17s: ' % i)
        line.append('precision: %6.2f%%; ' % (100.*m.prec))
        line.append('recall: %6.2f%%; ' % (100.*m.rec))
        line.append('FB1: %6.2f  %d\n' % (100.*m.fscore, c.tFoundGuessed[i]))
        finalReport.append("".join(line))
    return finalReport

def reportDict(counts, out=None):
    if out is None:
        out = sys.stdout
    overall, byType = metrics(counts)
    c = counts


    finalReportDict = {}
    finalReportDict['tokenCounter'] = c.tokenCounter
    finalReportDict['foundCorrect'] = c.foundCorrect
    finalReportDict['foundGuessed'] = c.foundGuessed
    finalReportDict['correctChunk'] = c.correctChunk
    if c.tokenCounter > 0: #整体统计
        wholeDict = {}
        wholeDict['Acc'] = 100.*c.correctTags/c.tokenCounter
        wholeDict['MacroPrecision'] = 100.*overall.prec
        wholeDict['MacroRecall'] = 100.*overall.rec
        wholeDict['MacroF1'] = 100.*overall.fscore
        finalReportDict['wholeClass'] = wholeDict
    subDict = {}
    for i, m in sorted(byType.items()):# 子类统计
        subDict[i] = {"Precision":100.*m.prec, "Recall":100.*m.rec, "F1":100.*m.fscore}
    finalReportDict['subClass'] = subDict

    # 以下直接输出
    finalReport = []
    line = []
    line.append('processed %d tokens with %d phrases; ' %
              (c.tokenCounter, c.foundCorrect))
    line.append('found: %d phrases; correct: %d.\n' %
              (c.foundGuessed, c.correctChunk))
    finalReport.append("".join(line))

    if c.tokenCounter > 0:
        line = []
        line.append('accuracy: %6.2f%%; ' %
                  (100.*c.correctTags/c.tokenCounter))
        line.append('precision: %6.2f%%; ' % (100.*overall.prec))
        line.append('recall: %6.2f%%; ' % (100.*overall.rec))
        line.append('FB1: %6.2f\n' % (100.*overall.fscore))
        finalReport.append("".join(line))

    for i, m in sorted(byType.items()):
        line = []
        line.append('%17s: ' % i)
        line.append('precision: %6.2f%%; ' % (100.*m.prec))
        line.append('recall: %6.2f%%; ' % (100.*m.rec))
        line.append('FB1: %6.2f  %d\n' % (100.*m.fscore, c.tFoundGuessed[i]))
        finalReport.append("".join(line))

    return finalReport, finalReportDict


def endOfChunk(prevTag, tag, prevType, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunkEnd = False

    if prevTag == 'E': chunkEnd = True
    if prevTag == 'S': chunkEnd = True

    if prevTag == 'B' and tag == 'B': chunkEnd  = True
    if prevTag == 'B' and tag == 'S': chunkEnd  = True
    if prevTag == 'B' and tag == 'O': chunkEnd  = True
    if prevTag == 'I' and tag == 'B': chunkEnd  = True
    if prevTag == 'I' and tag == 'S': chunkEnd  = True
    if prevTag == 'I' and tag == 'O': chunkEnd  = True

    if prevTag != 'O' and prevTag != '.' and prevType != type_:
        chunkEnd = True

    # these chunks are assumed to have length 1
    if prevTag == ']': chunkEnd = True
    if prevTag == '[': chunkEnd = True

    return chunkEnd


def startOfChunk(prevTag, tag, prevType, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunkStart = False

    if tag == 'B': chunkStart = True
    if tag == 'S': chunkStart = True

    if prevTag == 'E' and tag == 'E': chunkStart = True
    if prevTag == 'E' and tag == 'I': chunkStart = True
    if prevTag == 'S' and tag == 'E': chunkStart = True
    if prevTag == 'S' and tag == 'I': chunkStart = True
    if prevTag == 'O' and tag == 'E': chunkStart = True
    if prevTag == 'O' and tag == 'I': chunkStart = True

    if tag != 'O' and tag != '.' and prevType != type_:
        chunkStart = True

    # these chunks are assumed to have length 1
    if tag == '[': chunkStart = True
    if tag == ']': chunkStart = True

    return chunkStart


def returnReport(inputFile):
    with codecs.open(inputFile, "r", "utf8") as f:
        counts = evaluate(f)
    return reportDict(counts)


def main(argv):
    args = parseArgs(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)

if __name__ == '__main__':
    sys.exit(main(sys.argv))