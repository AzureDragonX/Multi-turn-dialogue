def Classifier(symptoms: list, symptom_list: dict) -> list:
    '''
    名称： 分类器
    作用： 接收症状实体，返回待筛疾病列表
    输入： 症状实体
    输出： 待筛疾病
    '''
    sicks = ()
    for sym in symptoms:
        if sicks == ():
            sicks = set(symptom_list[sym])
        else:
            sicks = sicks.intersection(set(symptom_list[sym]))

    

    return list(sicks)

if __name__ == "__main__":
    symptoms = ['注意减退', '兴奋']
    f = open("symptom.txt", 'r')
    all_dict = eval(f.read())
    f.close()
    sicks = Classifier(symptoms, all_dict)
    print(sicks)

    


