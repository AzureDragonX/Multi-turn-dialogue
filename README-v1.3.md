# 多轮问答接口文档



### 1.首轮对话

#### 1-1.首轮对话请求参数

```python
{
    /* 会话或用户id */
    'id': "8589a53c-022d-4fcf-8b90-97875da145ee"，
    
    /* 当前对话轮次 */
    'cur': '0',
    
    /* 对话历史 */
    'dialogue':
    [
        {
        /* 对话轮次 */
        'turn': '0', 
        /* 输入 */
        'inputs': ('Request', 'Diagnosis', '我有点腹泻、抑郁、气促'),
        /* 输出 */
        'outputs':()
        }
    ]
}

```

备注：需传入完整的JSON，其中user_input即为用户输入的自述症状。



#### 1-2.返回数据

```python
{
    'id': '12345', 
    'cur': '0', 
    'dialogue': 
    [
        {
            'turn': '0', 
            'input': ('Request', 'Diagnosis', '你好，我有腹泻、抑郁、气促'),
            'output': ('Request', 'Symptom', '呕吐')
        }
    ], 
    
    /* 待筛疾病列表 */
    'disease': 
    [
        {
             '呕吐': None,
             '上腹部疼痛': None, 
             '剧烈腹痛': None, 
             '腹泻': True, 
             '抑郁': True, 
             '焦虑和紧张': None, 
             '视力下降': None, 
             '口腔溃疡': None, 
             '咳嗽': None, 
             '脖子痛': None, 
             '失明': None, 
             '气促': True, 
             'state': True, 
             'name': '白塞病'
        }
    ]
}
```

备注：此返回JSON在后续多轮问答中需要前后端共同维护。前端可根据当前轮次的'output'内容决定输出问句。此例中，'output'为('Request', 'Symptom', '呕吐')，意思是需要向用户问询是否有'呕吐'这个症状。



### 2.后续轮次对话

#### 2-1.后续对话请求参数

在接受到用户回答后，需要将回答存入对话历史的新轮次中，并使轮次数'cur'自加1，保证一致，而**其他数据仍需原样输入**到算法中。例如，当用户回答“有的”之后，传入参数就变为：

```python
{
    'id': '12345', 
    
    /* 轮次加一 */
    'cur': '1', 
    
    'dialogue': 
    [
        {
            'turn': '0', 
            'input': ('Request', 'Diagnosis', '你好，我有腹泻、抑郁、气促'),
            'output': ('Request', 'Symptom', '呕吐')
        },
        
        /* 新增一轮，其余内容不变 */
        {
            'turn': '1', 
            'input': ('Inform', 'Answer', '有的'),
            'output': ()
        }
    ], 
    'disease': 
    [
        {
             '呕吐': None,
             '上腹部疼痛': None, 
             '剧烈腹痛': None, 
             '腹泻': True, 
             '抑郁': True, 
             '焦虑和紧张': None, 
             '视力下降': None, 
             '口腔溃疡': None, 
             '咳嗽': None, 
             '脖子痛': None, 
             '失明': None, 
             '气促': True, 
             'state': True, 
             'name': '白塞病'
        }
    ]
}
```



#### 2-2.返回数据

形式同首轮，算法主要工作就是调整待筛疾病列表以及填充当前轮次对话的'output'槽。





### 3.Act-Slot 元组

在共同维护的JSON数据中，用户意图(input)和算法意图(output)都是用Act-Slot的方式归一的，这也是方便以后能够随时添加新的各种各样的意图。在目前的系统中，只有4种Act-Slot 元组。

| 元组                                 | 功能                                                         | 备注                                                         |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ('Request', 'Diagnosis', user_input) | 请求诊断——触发对话，暂仅用于首轮对话的input中                | 用户输入放在user_input中                                     |
| ('Request', 'Symptom', symp_name)    | 询问症状——系统下一轮想要向用户询问的症状，一般放在对话中的output | 症状名存放在symp_name中                                      |
| ('Inform', 'Answer', user_input)     | 回答问询——用户对系统发问的症状的回答，一般放在对话中的input  | 用户输入放在user_input中                                     |
| ('Inform', 'Disease', disease)       | 结束诊断——预测或拒绝预测，中止对话，一般放在最后一轮对话的output中 | disease若为字符串，则说明成功预测某个疾病；若为None，则说明拒绝预测。 |

