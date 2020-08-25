
affirm = ['有啊','有！', '有!', '有','有的','应该有','是','是的','确定','非常确定','一定','一定是','肯定','肯定是','肯定有','一定有','确实是','没错','对','对的','嗯','嗯嗯','恩','yes','yeah','yep','系呀','系','不错']
sick1_db = ['咳嗽','气促','鼻塞','发烧','喉咙痛','呼吸困难','胸痛','鼻炎','哮喘','咳痰','胸部充血','胸闷']
sick2_db = ['咳嗽','发烧','鼻塞','鼻炎','喉咙痛','气促','呼吸困难','头痛','耳朵痛','哮喘']
sick3_db = ['胸痛','气促','咳嗽','呼吸困难','呼吸痛','背痛','肋骨痛','发烧','腹痛']
sick4_db = ['胸痛','气促','肩痛','呼吸困难','肋骨痛','背痛','咳嗽','咳痰','呼吸痛']
sick5_db = ['气促','呼吸困难','胸痛','咳嗽','哮喘','水肿','窒息','呼吸痛','心悸','咯血']

def Validator(sick: str, slots: dict, flag: bool = False):
    '''
    名称： 确认器
    作用： 通过多轮问答形式，确认是否触发该疾病确认信息
    输入： 疾病 slot 字典
    输出： True: 确认信息 False: 返回调度器
    '''
    # 慢阻肺验证
    if sick == '慢性阻塞性肺疾病（COPD）':

        slot1 = {i:j for i, j in slots.items() if i in sick1_db}
        confidence = 0

        if slot1['胸部充血'] == False:
            print('请问您有胸部充血症状吗？')
            answer = input()
            if answer in affirm:
                slots['胸部充血'] = True
                slot1['胸部充血'] = True
            else:
                return 0

        if slot1['胸闷'] == False:
            print('请问您有胸闷症状吗？')
            answer = input()
            if answer in affirm:
                slots['胸闷'] = True
                slot1['胸闷'] = True
            else:
                return 0

        for a, b in slot1.items():
            if b == True:
                confidence += 1
        
        if confidence >= 6:
            print('根据您的描述，推测您可能的疾病是慢性阻塞性肺疾病（COPD），结果仅供参考，建议你前往附近的医院做进一步检查。')
            flag = True
            return 0
                
        for m,n in slot1.items():

            if n != True:
                print('请问您有'+m+'症状吗？')
                answer = input()
                if answer in affirm:
                    slots[m] = True
                    slot1[m] = True
                    confidence +=1
                
            if confidence>=6:
                print('根据您的描述，推测您可能的疾病是慢性阻塞性肺疾病（COPD），结果仅供参考，建议你前往附近的医院做进一步检查。')
                flag =True
                break
        return 0

    # 间质性肺病验证  
    if sick == '间质性肺病':

        slot2 = {i:j for i, j in slots.items() if i in sick2_db}
        confidence = 0

        if slot2['耳朵痛'] == False:
            print('请问您有耳朵痛症状吗？')
            answer = input()
            if answer in affirm:
                slots['耳朵痛'] = True
                slot2['耳朵痛'] = True
            else:
                return 0

        if slot2['头痛'] == False:
            print('请问您有头痛吗？')
            answer = input()
            if answer in affirm:
                slots['头痛'] = True
                slot2['头痛'] = True
            else:
                return 0

        for a, b in slot2.items():
            if b == True:
                confidence += 1
        
        if confidence >= 5:
            print('根据您的描述，推测您可能的疾病是间质性肺病，结果仅供参考，建议你前往附近的医院做进一步检查。')
            flag = True
            return 0
                
        for m,n in slot2.items():

            if n != True:
                print('请问您有'+m+'症状吗？')
                answer = input()
                if answer in affirm:
                    slots[m] = True
                    slot2[m] = True
                    confidence +=1
                
            if confidence>=6:
                print('根据您的描述，推测您可能的疾病是间质性肺病，结果仅供参考，建议你前往附近的医院做进一步检查。')
                flag =True
                break
        return 0

    # 胸腔积液验证  
    if sick == '胸腔积液':

        slot3 = {i:j for i, j in slots.items() if i in sick3_db}
        confidence = 0

        if slot3['腹痛'] == False:
            print('请问您有腹痛症状吗？')
            answer = input()
            if answer in affirm:
                slots['腹痛'] = True
                slot3['腹痛'] = True
            else:
                return 0


        for a, b in slot3.items():
            if b == True:
                confidence += 1
        
        if confidence >= 5:
            print('根据您的描述，推测您可能的疾病是胸腔积液，结果仅供参考，建议你前往附近的医院做进一步检查。')
            flag = True
            return 0
                
        for m,n in slot3.items():

            if n != True:
                print('请问您有'+m+'症状吗？')
                answer = input()
                if answer in affirm:
                    slots[m] = True
                    slot3[m] = True
                    confidence +=1
                
            if confidence>=5:
                print('根据您的描述，推测您可能的疾病是胸腔积液，结果仅供参考，建议你前往附近的医院做进一步检查。')
                flag =True
                break
        return 0

    # 气胸验证  
    if sick == '气胸':

        slot4 = {i:j for i, j in slots.items() if i in sick4_db}
        confidence = 0

        if slot4['肩痛'] == False:
            print('请问您有肩痛症状吗？')
            answer = input()
            if answer in affirm:
                slots['肩痛'] = True
                slot4['肩痛'] = True
            else:
                return 0


        for a, b in slot4.items():
            if b == True:
                confidence += 1
        
        if confidence >= 5:
            print('根据您的描述，推测您可能的疾病是气胸，结果仅供参考，建议你前往附近的医院做进一步检查。')
            flag = True
            return 0
                
        for m,n in slot4.items():

            if n != True:
                print('请问您有'+m+'症状吗？')
                answer = input()
                if answer in affirm:
                    slots[m] = True
                    slot4[m] = True
                    confidence +=1
                
            if confidence>=5:
                print('根据您的描述，推测您可能的疾病是胸腔积液，结果仅供参考，建议你前往附近的医院做进一步检查。')
                flag =True
                break
        return 0
    
    # 肺动脉高压验证  
    if sick == '肺动脉高压':

        slot5 = {i:j for i, j in slots.items() if i in sick5_db}
        confidence = 0

        if slot5['心悸'] == False:
            print('请问您有心悸症状吗？')
            answer = input()
            if answer in affirm:
                slots['心悸'] = True
                slot5['心悸'] = True
            else:
                return 0

        if slot5['咯血'] == False:
            print('请问您有咯血症状吗？')
            answer = input()
            if answer in affirm:
                slots['咯血'] = True
                slot5['咯血'] = True
            else:
                return 0
        
        if slot5['水肿'] == False:
            print('请问您有水肿症状吗？')
            answer = input()
            if answer in affirm:
                slots['水肿'] = True
                slot5['水肿'] = True
            else:
                return 0
        
        if slot5['窒息'] == False:
            print('请问您有心悸症状吗？')
            answer = input()
            if answer in affirm:
                slots['窒息'] = True
                slot5['窒息'] = True
            else:
                return 0
        
        for a, b in slot5.items():
            if b == True:
                confidence += 1
        
        if confidence >= 5:
            print('根据您的描述，推测您可能的疾病是肺动脉高压，结果仅供参考，建议你前往附近的医院做进一步检查。')
            flag = True
            return 0
                
        for m,n in slot5.items():

            if n != True:
                print('请问您有'+m+'症状吗？')
                answer = input()
                if answer in affirm:
                    slots[m] = True
                    slot5[m] = True
                    confidence +=1
                
            if confidence>=5:
                print('根据您的描述，推测您可能的疾病是肺动脉高压，结果仅供参考，建议你前往附近的医院做进一步检查。')
                flag =True
                break
        return 0
    

       
      
                      
                      

                         
                   

