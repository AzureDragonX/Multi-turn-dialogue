from utils_main.classifier import Classifier
from utils_main.dispatcher import Dispatcher
import os
import warnings
from rasa.nlu.model import Interpreter
from utils_main.validator import Validator
import pandas as pd

# 载入症状——疾病权重字典
f = open("utils_main//symptom.txt", 'r')
all_dict = eval(f.read())
f.close()

# id_symptom是症状——id对应表
temp = pd.read_csv('id_symptom.csv')
id_symptom = {}
for tup in zip(temp['id'], temp['symptom']):
    id_symptom.update({tup[0]: tup[1]})


def main():
    '''
    多轮问答主程序
    接收用户输入
    通过多轮问答形式，返回最后输出
    '''
    # model = model.Metadata.load('./nlu_models')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')
    # pipeline = ['HFTransformersNLP','LanguageModelTokenizer','LanguageModelFeaturizer','EntitySynonymMapper','CRFEntityExtractor','SklearnIntentClassifier']
    interpreter = Interpreter.load(model_dir='./rasa_nlu/nlu')
    print('您好，很高兴为您服务')
    print('能向我描述您的症状吗？最好提供三个或三个以上的症状哦')
    user_input = input()

    symptoms = []
    entities = interpreter.parse(user_input)['entities']
    for entity in entities:
        symptoms.append(id_symptom[entity['entity']])

    sicks = Classifier(symptoms, all_dict)
    print(sicks, symptoms)
    if sicks == []:
        print('根据您的描述，暂时无法确认您的疾病，建议您前往附近的医院做进一步检查')
        print('感谢您的使用，再见')
        return 0
    
    Dispatcher(sicks, symptoms)

    
    
    print('感谢您的使用，再见')

if __name__ == "__main__":
    main()
