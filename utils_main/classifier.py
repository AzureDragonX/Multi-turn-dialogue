sick1_db = ['咳嗽','气促','鼻塞','发烧','喉咙痛','呼吸困难','胸痛','鼻炎','哮喘','咳痰','胸部充血','胸闷']
sick2_db = ['咳嗽','发烧','鼻塞','鼻炎','喉咙痛','气促','呼吸困难','头痛','耳朵痛','哮喘']
sick3_db = ['胸痛','气促','咳嗽','呼吸困难','呼吸痛','背痛','肋骨痛','发烧','腹痛']
sick4_db = ['胸痛','气促','肩痛','呼吸困难','肋骨痛','背痛','咳嗽','咳痰','呼吸痛']
sick5_db = ['气促','呼吸困难','胸痛','咳嗽','哮喘','水肿','窒息','呼吸痛','心悸','咯血']


def Classifier(symptoms: list)-> list:
    '''
    名称： 分类器
    作用： 接收症状实体，返回待筛疾病
    输入： 症状实体
    输出： 待筛疾病
    '''
    sicks = []
    if set(symptoms).issubset(set(sick1_db)):
        sicks.append('慢性阻塞性肺疾病（COPD）')

    if set(symptoms).issubset(set(sick2_db)):
        sicks.append('间质性肺病')

    if set(symptoms).issubset(set(sick3_db)):
        sicks.append('胸腔积液')
    
    if set(symptoms).issubset(set(sick4_db)):
        sicks.append('气胸')
    
    if set(symptoms).issubset(set(sick5_db)):
        sicks.append('肺动脉高压')
    

    return sicks

if __name__ == "__main__":
    symptoms = ['咳嗽','发烧']
    sicks = Classifier(symptoms)
    print(sicks)

    


