#import rasa
from  rasa.nlu.model import Interpreter
#model = model.Metadata.load('./nlu_models')
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')
#pipeline = ['HFTransformersNLP','LanguageModelTokenizer','LanguageModelFeaturizer','EntitySynonymMapper','CRFEntityExtractor','SklearnIntentClassifier']
interpreter = Interpreter.load(model_dir= './nlu')
Flag = True
while Flag:
    inputs = input()
    if inputs == 'stop':
        Flag = False
    output = interpreter.parse(inputs)
    print(output)
    
    



