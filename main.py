from utils.classifier import Classifier
from utils.dispatcher import Dispatcher
from utils.validator import Validator
#import readline
def main():
    '''
    多轮问答主程序
    接收用户输入
    通过多轮问答形式，返回最后输出
    '''

    print('您好，很高兴为您服务')
    print('能向我描述您的症状吗？最好提供三个或三个的症状哦')

    user_input = input()
    '''
    NLU(user_input)
    Classifier
    Dispatcher
    Validator
    ....
    '''
    print('感谢您的使用，再见')

if __name__ == "__main__":
    main()
