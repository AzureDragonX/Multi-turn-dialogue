from utils_main.classifier import Classifier
from utils_main.dispatcher import Dispatcher
from utils_main.validator import Validator
from Align import return_symptom_list
from BilstmCrf import one_line


sickall_db = ['胸闷', '头痛', '耳朵痛', '水肿', '咯血', '心悸', '呼吸困难', '窒息', '鼻炎', '咳痰', '腹痛', '胸痛', '呼吸痛', '肋骨痛', '肩痛', '喉咙痛', '气促', '鼻塞', '咳嗽', '背痛', '胸部充血', '发烧', '哮喘']
SLOTS = {'胸闷': False, '头痛': False, '耳朵痛': False, '水肿': False, '咯血': False, '心悸': False, '呼吸困难': False, '窒息': False, '鼻炎': False, '咳痰': False, '腹痛': False, '胸痛': False, '呼吸痛': False, '肋骨痛': False, '肩痛': False, '喉咙痛': False, '气促': False, '鼻塞': False, '咳嗽': False, '背痛': False, '胸部充血': False, '发烧': False, '哮喘': False}
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

    for symptom in symptoms:
        SLOTS[symptom] = True

    SICKS = Classifier(symptoms)
    if SICKS == []:
        print('根据您的描述，暂时无法确认您的疾病，建议您前往附近的医院做进一步检查')
        print('感谢您的使用，再见')
        return 0
    
    Dispatcher(SICKS,SLOTS)
    
    #Dispatcher
    #Validator
    
    
    print('感谢您的使用，再见')

if __name__ == "__main__":
    main()
