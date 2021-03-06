from .validator import Validator


def Dispatcher(sicks: list, symptoms: list):
    '''
    名称： 调度器
    作用： 管理待筛疾病，将其有序分发给具体的疾病确认器
    输入： 待筛疾病
    输出： 具体疾病确认器
    '''
    flag = False
    for sick in sicks:
        if not flag:
            flag, symptoms = Validator(sick, symptoms)
            if flag:
                break

    if not flag:
        print('根据您的描述，暂时无法确认您的疾病，建议您前往附近的医院做进一步检查')


if __name__ == "__main__":
    sick1 = [1]
    Dispatcher(sick1)
