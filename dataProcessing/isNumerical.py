# 将class 的值转为数值类型
import numpy as np

# 传进来的y必须是 类那一列的数据
# 如果是字符类型的，那就把该数据转化成相对应的 int类型 {0，1，2.....}
def reserveNumerical(y):
    # 判断数据是否是str
    if isinstance(y[0], str):
        valueOfClass = y
        tempy = np.asarray([])
        m = set(valueOfClass[i] for i in range(valueOfClass.shape[0]))
        m = list(m)
        m = np.asarray(m)
        y_value = np.arange(0, len(m))
        for i in range(0, len(valueOfClass)):
            index = np.where(m == valueOfClass[i])[0][0]
            tempy = np.append(tempy, y_value[index])
        y = tempy
    return y