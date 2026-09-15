import numpy as np
# 将class 的值转为数值类型
def reserveNumerical(dataY):
    # 判断数据是否是str
    y = dataY
    if isinstance(y[0], str):
        valueOfClass = y
        tempy = np.asarray([])
        m = set(valueOfClass[i] for i in range(valueOfClass.shape[0]))
        # set要先转list才能转array
        m = list(m)
        m = np.asarray(m)
        y_value = np.arange(0, len(m))
        for i in range(0, len(valueOfClass)):
            index = np.where(m == valueOfClass[i])[0][0]
            tempy = np.append(tempy, y_value[index])
        y = tempy
    dataY = y
    return dataY


def strToNum(dataY):
    y = dataY
    valueOfClass = y
    tempy = np.asarray([])
    m = set(valueOfClass[i] for i in range(valueOfClass.shape[0]))
    # set要先转list才能转array
    m = list(m)
    m = np.asarray(m)
    y_value = np.arange(0, len(m))
    for i in range(0, len(valueOfClass)):
        index = np.where(m == valueOfClass[i])[0][0]
        tempy = np.append(tempy, y_value[index])
    y = tempy
    return y

