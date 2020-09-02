import pandas as pd
dis = pd.read_csv('disease-symptom.csv')

affirm = ['有啊','有！', '有!', '有','有的','应该有','是','是的','确定','非常确定','一定','一定是','肯定','肯定是','肯定有',
          '一定有','确实是','没错','对','对的','嗯','嗯嗯','恩','yes','yeah','yep','系呀','系','不错']



def Validator(sick: str, symptoms: list):
    '''
    名称： 确认器
    作用： 通过多轮问答形式，确认是否触发该疾病确认信息
    输入： 疾病名称 症状列表
    输出： True: 确诊 False: 无法确诊
    '''
    temp = dis.loc[dis['疾病名称'] == sick]
    temp.sort_values(by='权重分数', axis=0, ascending=False, inplace=True)
    symp_list = list(temp['标准临床表现'])
    score_list = list(temp['权重分数'])
    thres = sum(score_list)/2
    score = sum([score_list[symp_list.index(i)] for i in symptoms])
    left_score = sum(score_list)-score

    if score >= thres:
        print('我们认为您很可能患上了{0}'.format(sick))
        return True, symptoms
    for symp in symp_list:
        if symp in symptoms:
            continue
        left_score -= score_list[symp_list.index(symp)]
        print('请问您有'+symp+'症状吗？')
        answer = input()
        if answer in affirm:
            symptoms.append(symp)
            score += score_list[symp_list.index(symp)]
        elif score_list[symp_list.index(symp)] > 15 or score + left_score < thres:
            return False, symptoms

        if score >= thres:
            print('我们认为您很可能患上了{0}'.format(sick))
            return True, symptoms


if __name__ == '__main__':
    print(Validator('多发性硬化', ['注意减退', '兴奋']))

    

       
      
                      
                      

                         
                   

