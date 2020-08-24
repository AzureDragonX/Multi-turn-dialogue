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

    


