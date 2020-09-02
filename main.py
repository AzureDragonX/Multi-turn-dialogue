from utils_main.classifier import Classifier
from utils_main.dispatcher import Dispatcher
from utils_main.validator import Validator
from Align import return_symptom_list
from BilstmCrf import one_line
import pandas as pd

# 载入症状——疾病权重字典
f = open("utils_main//symptom.txt", 'r')
all_dict = eval(f.read())
f.close()


def main():
    '''
    多轮问答主程序
    接收用户输入
    通过多轮问答形式，返回最后输出
    '''
    print('您好，很高兴为您服务')
    print('能向我描述您的症状吗？最好提供三个或三个以上的症状哦')
    user_input = input()
    
    symptoms =  return_symptom_list(one_line(user_input))

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
