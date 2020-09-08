import pandas as pd

# disease-symptom是疾病症状总表
dis_df = pd.read_csv('data//disease-symptom.csv')


def Validator(disease_dict):
    '''
    名称： 确认器
    作用： 通过多轮问答形式，确认是否触发该疾病确认信息
    输入： 疾病字典，包含所有症状以及name、state
    输出： 下一个要问的症状，特殊标识：'No'代表换病，'Yes'代表确诊此病。
    '''
    # 已经确认的症状先收集起来
    symptoms = [k for k, v in disease_dict.items() if k != 'name' and k != 'state' and v is True]

    # 已经确认不是的症状也收集起来
    no_symptoms = [k for k, v in disease_dict.items() if k != 'name' and k != 'state' and v is False]

    # 拿到疾病的子表并对权值排序
    temp = dis_df.loc[dis_df['disease'] == disease_dict['name']]
    temp.sort_values(by='weight', axis=0, ascending=False, inplace=True)

    # 该病排序号的症状表、权值表
    symp_list = list(temp['symptom'])
    score_list = list(temp['weight'])

    thres = sum(score_list) / 2
    score = sum([score_list[symp_list.index(i)] for i in symptoms])
    no_score = sum([score_list[symp_list.index(i)] for i in no_symptoms])

    if sum(score_list) - no_score + score < thres:
        return 'No'
    if score >= thres:
        return 'Yes'
    for symp in symp_list:
        if disease_dict[symp] == None:
            return symp

    return 'No'


if __name__ == '__main__':
    print(Validator('多发性硬化', ['注意减退', '兴奋']))
