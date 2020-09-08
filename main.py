from utils_main.classifier import Classifier
from utils_main.validator import Validator
import pandas as pd
import ahocorasick
from afw.OutputStandard import normalTable

affirm = ['也有', '确实有', '有啊','有！', '有!', '有','有的','应该有','是','是的','确定','非常确定','一定','一定是','肯定','肯定是','肯定有',
          '一定有','确实是','没错','对','对的','嗯','嗯嗯','恩','yes','yeah','yep','系呀','系','不错']

# ('Request', 'Diagnosis', input) - 用户请求诊断,input为自述信息
# ('Inform', 'Answer', input) - 用户对症状回答信息
# ('Request', 'Symptom', output) - 询问用户症状
# ('Inform', 'Disease', output) - 确诊某病，None则说明无法确诊

def build_actree(wordlist):
    actree = ahocorasick.Automaton() #建空树
    for index, word in enumerate(wordlist):
        actree.add_word(word, (index, word)) #向trie树中添加单词
    actree.make_automaton() #将trie树转化为Aho-Corasick自动机
    return actree


# 载入症状——疾病字典
f = open("data//symptom.txt", 'r')
sym2dis_dict = eval(f.read())
f.close()

sym_list = sym2dis_dict.keys()
sym_tree = build_actree(sym_list)

# 4列分别为disease, symptom, weight, strong
dis_df = pd.read_csv('data//disease-symptom.csv')


def main(inputData):
    '''
    多轮问答主程序
    接收用户输入
    通过多轮问答形式，返回最后输出
    '''

    cur = inputData['cur']
    cur_dia = inputData['dialogue'][int(cur)]
    assert cur == cur_dia['turn']

    if cur == '0':
        symptoms = []
        for i in sym_tree.iter(cur_dia['input'][2]):  # 找到所有匹配的字符串，返回一个iterator
            wd = i[1][1]  # i的格式：(节点index, (字符串序号，字符串))
            symptoms.append(wd)
        diseases = Classifier(symptoms, sym2dis_dict)
        inputData.update({'disease': []})
        for dis in diseases:
            temp = dis_df.loc[dis_df['disease'] == dis]
            temp_dict = {}
            for tup in temp.itertuples():
                if tup[2] in symptoms:
                    temp_dict.update({tup[2]: True})
                else:
                    temp_dict.update({tup[2]: None})
            temp_dict.update({'state': None})
            temp_dict.update({'name': dis})
            inputData['disease'].append(temp_dict)


        # 查一下是否已经有确诊的疾病，有就直接返回，否则就走正常流程，取疾病列表第一个开问
        temp = []
        for item in inputData['disease']:
            temp.append(Validator(item))
        if 'Yes' not in temp:
            symp = Validator(inputData['disease'][0])
            inputData['disease'][0]['state'] = True
            cur_dia['output'] = ('Request', 'Symptom', symp)
            inputData['dialogue'][int(cur)] = cur_dia
            return inputData
        else:
            cur_dia['output'] = ('Inform', 'Disease', inputData['disease'][temp.index('Yes')]['name'])
            inputData['dialogue'][int(cur)] = cur_dia
            return inputData

    else:
        answer = cur_dia['input'][2]
        asking_dis = -1
        for idx, i in enumerate(inputData['disease']):
            if i['state']:
                asking_dis = idx
                break

        # 确认是否是肯定回答，并调整在问疾病的在问症状的值，output中含的是一个三元组，第三个元素是上一轮询问的症状
        prev_symp = inputData['dialogue'][int(cur)-1]['output'][2]
        if answer in affirm:
            for i in inputData['disease']:
                if prev_symp in i.keys():
                    i[prev_symp] = True
        else:
            for i in inputData['disease']:
                if prev_symp in i.keys():
                    i[prev_symp] = False
        symp = Validator(inputData['disease'][asking_dis])

        # 分别对确诊、换病、询问下一个症状完成相应处理
        if symp == 'Yes':
            # 确诊，返回确诊动作
            cur_dia['output'] = ('Inform', 'Disease', inputData['disease'][asking_dis]['name'])
            inputData['dialogue'][int(cur)] = cur_dia
            return inputData
        elif symp == 'No':
            # 换病
            symp_ = symp
            while asking_dis < len(inputData['disease']) - 1 and symp_ == 'No':
                inputData['disease'][asking_dis]['state'] = False
                inputData['disease'][asking_dis + 1]['state'] = True
                symp_ = Validator(inputData['disease'][asking_dis + 1])
                asking_dis += 1

            # 看是否已经问完了，没有的话下一个病开问
            if asking_dis == len(inputData['disease']) - 1:
                cur_dia['output'] = ('Inform', 'Disease', None)
                inputData['dialogue'][int(cur)] = cur_dia
                return inputData
            else:
                cur_dia['output'] = ('Request', 'Symptom', symp_)
                inputData['dialogue'][int(cur)] = cur_dia
                return inputData

        else:
            cur_dia['output'] = ('Request', 'Symptom', symp)
            inputData['dialogue'][int(cur)] = cur_dia
            return inputData


if __name__ == "__main__":
    print('您好，很高兴为您服务')
    print('能向我描述您的症状吗？最好提供三个或三个以上的症状哦')
    user_input = input()
    inputData = {
                    'id': '12345',
                    'cur': '0',
                    'dialogue': [{'turn': '0', 'input': ('Request', 'Diagnosis', user_input), 'output': ()}],
                }
    result = main(inputData)
    print(result)
    while result['dialogue'][int(result['cur'])]['output'][0] != 'Inform':
        print('请问你有'+result['dialogue'][int(result['cur'])]['output'][2]+'吗？')
        user_input = input()
        cur = int(result['cur'])+1
        result['dialogue'].append({'turn': cur, 'input': ('Inform', 'Answer', user_input), 'output': ()})
        result['cur'] = cur
        result = main(result)
        print(result)
    if result['dialogue'][int(result['cur'])]['output'][2]:
        print('我们认为你可能患上了'+result['dialogue'][int(result['cur'])]['output'][2])
    else:
        print('无法确诊，请寻求其他方式救助。')

