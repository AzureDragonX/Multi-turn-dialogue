from utils.classifier import Classifier
from utils.dispatcher import Dispatcher
from utils.validator import Validator

sick1_db = ['咳嗽','气促','鼻塞','发烧','喉咙痛','呼吸困难','胸痛','鼻炎','哮喘','咳痰','胸部充血','胸闷']
sick2_db = ['咳嗽','发烧','鼻塞','鼻炎','喉咙痛','气促','呼吸困难','头痛','耳朵痛','哮喘']
sick3_db = ['胸痛','气促','咳嗽','呼吸困难','呼吸痛','背痛','肋骨痛','发烧','腹痛']
sick4_db = ['胸痛','气促','肩痛','呼吸困难','肋骨痛','背痛','咳嗽','咳痰','呼吸痛']
sick5_db = ['气促','呼吸困难','胸痛','咳嗽','哮喘','水肿','窒息','呼吸痛','心悸','咯血']
sickall_db = ['胸闷', '头痛', '耳朵痛', '水肿', '咯血', '心悸', '呼吸困难', '窒息', '鼻炎', '咳痰', '腹痛', '胸痛', '呼吸痛', '肋骨痛', '肩痛', '喉咙痛', '气促', '鼻塞', '咳嗽', '背痛', '胸部充血', '发烧', '哮喘']
SLOTS = {'胸闷': False, '头痛': False, '耳朵痛': False, '水肿': False, '咯血': False, '心悸': False, '呼吸困难': False, '窒息': False, '鼻炎': False, '咳痰': False, '腹痛': False, '胸痛': False, '呼吸痛': False, '肋骨痛': False, '肩痛': False, '喉咙痛': False, '气促': False, '鼻塞': False, '咳嗽': False, '背痛': False, '胸部充血': False, '发烧': False, '哮喘': False}
affirm = ['有','有的','应该有','是','是的','确定','非常确定','一定','一定是','肯定','肯定是','肯定有','一定有','确实是','没错','对','对的','嗯','嗯嗯','恩','yes','yeah','yep','系呀','系','不错']
def main():
    '''
    多轮问答主程序
    接收用户输入
    通过多轮问答形式，返回最后输出
    '''
    print('您好，很高兴为您服务')
    print('能向我描述您的症状吗？最好提供三个或三个的症状哦')
    user_input = input()
    
    #symptoms =  NLU(user_input)

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
