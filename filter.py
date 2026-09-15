
import numpy as np

from dataProcessing.InforGain import InforGain
# from dataProcessing.reliefF import reliefFScore
from dataProcessing.reliefF import reliefFScore
# from dataProcessing.newreliefF import reliefFScore
# from dataProcessing.KneePointDivideData import findKneePoint
from skfeature.utility.mutual_information import information_gain, su_calculation


class FilterOfData:
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 保存relifF的值
    scoreOfRelifF = np.asarray([])
    # 保存了信息增益
    inforGainArray = np.asarray([])
    # 保存了对称不确定SU
    featureClassSU = np.asarray([])

    # 初始化
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY

    #计算relifF的值
    def computerRelifFScore(self):
        self.scoreOfRelifF = reliefFScore(self.dataX,self.dataY)
        return self.scoreOfRelifF

    # 计算互信息的值
    def computerInforGain(self):
        length = len(self.dataX[0])
        self.inforGainArray = np.zeros(length)
        f1 = self.dataY
        import time
        start = time.time()
        for feature_i in range(0, length):
            f2 = self.dataX[:, feature_i]
            self.inforGainArray[feature_i] = information_gain(f1, f2)
        print(f"mutualInforamation time = {time.time() - start} seconds")
        return self.inforGainArray

    # 得到su矩阵  feature ---class
    def getFeatureClassSU(self):
        length = self.dataX.shape[1]
        import time
        start = time.time()
        self.featureClassSU = np.zeros(length)
        f1 = self.dataY
        for feature_i in range(0, length - 1):
            f2 = self.dataX[:, feature_i]
            self.featureClassSU[feature_i] = su_calculation(f1, f2)
        print(f"SU time = {time.time() - start} seconds")
        return self.featureClassSU


    # filter 过滤
    def filter_relifF(self,scoreOfRelifF,dataX,dataFeature):
        # 拐点分割特征，选择拐点前的数据
        if len(dataFeature) > 1000:
            # extratRatio = findKneePoint(data=self.scoreOfRelifF)
            extratRatio = 0.03
        else:
            extratRatio = 1
        dataLen = dataX.shape[1]
        exR = int(dataLen * extratRatio)
        # 降序排序 relifF评分
        indexOfRelifF = np.argsort(scoreOfRelifF)[::-1]
        tenpI = indexOfRelifF[:exR]
        scoreOfRelifF = scoreOfRelifF[tenpI]
        dataX = dataX[:, tenpI]
        dataFeature = dataFeature[tenpI]
        return scoreOfRelifF,dataX,dataFeature

    def filter_relifF_test(self,scoreOfRelifF,dataX,dataFeature):
        # 拐点分割特征，选择拐点前的数据
        if len(dataFeature) > 1000:
            # extratRatio = findKneePoint(data=self.scoreOfRelifF)
            extratRatio = 0.03
        else:
            extratRatio = 1
        dataLen = dataX.shape[1]
        exR = int(dataLen * extratRatio)
        # 降序排序 relifF评分
        indexOfRelifF = np.argsort(scoreOfRelifF)[::-1]
        tenpI = indexOfRelifF[:exR]
        dataX = dataX[:, tenpI]
        return dataX