#  ------ REMODE -------#

import os
import numpy as np
import openpyxl
import pandas as pd
from random import sample

from numpy import asarray
from scipy.stats import cauchy
from win32con import NULLREGION
from win32verstamp import nullterm

from Filter.filter import FilterOfData
from dataProcessing.ReadDataCSV_new import ReadCSV
from dataProcessing.NonDominate import nonDominatedSort_PFAndPop, getPF
from sklearn.preprocessing import MinMaxScaler
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV
#from sklearn.model_selection import train_test_split
# from dataProcessing.drawPlo import pltForWin


class Genetic:
    # 保存全局数据
    dataX_global = np.asarray([])
    # 保存了数据中除了class意外的其他数据
    dataX = np.asarray([])
    # 保存了类数据
    dataY = np.asarray([])
    # 测试集
    X_test = np.asarray([])
    Y_test = np.asarray([])
    # 特征的存放数组
    dataFeature = np.asarray([])
    # 数据的全部索引
    allDataIndex = np.asarray([])
    # 特征的数量
    dataFeatureNum = 0
    # 每一个特征的relifF评分
    scoreOfRelifF = np.asarray([])
    # 将种群中的solution提取出来
    solutionArray = []
    # 种群
    population_DE = np.asarray([])
    # 保存 在选择算子中被淘汰的个体
    inferiorPop = np.asarray([])
    # 全局档案
    globalArchive = np.asarray([])
    # 局部档案
    localArchive = np.asarray([])
    # 每轮递归保留的个体档案
    saveArchive = np.asarray([])
    # 每一个种群的数量
    populationNum = 0
    # 种群迭代次数
    iteratorTime = 0
    # 全局保存特征数量
    globalFN = 0
    # 保存全局的数据
    globalDI = np.asarray([])
    # 精英种群通过fitness排序得到的原始前 p% 个个体得到
    p = 0
    # 实数 转 二进制
    eta = 0
    # c
    c = 0
    # 提取的最优个体数量
    bm = 0
    # 默认的自适应的F参数
    mu_F = 0
    # 默认的自适应的crossoverPro参数
    mu_CR = 0
    CV = 0
    # 全局最优解
    globalSolution = np.asarray([])
    # 全局最优解的实数制
    globalSolutionNum = np.asarray([])
    # 全局最优解的适应度值
    globalFitness = 1
    # 全局最优的score
    globalScore = 0
    # 是否继续递归
    isRecursive = True
    # 保存数据名字
    dataName = ""
    globalFitArray = []
    sorted_feature = []


    # initialization function
    def __init__(self,dataX,dataY,dataName):
        # self.dataX, self.X_test, self.dataY, self.Y_test = train_test_split(dataX, dataY, test_size=0.3, stratify=dataY)
        self.dataX = dataX
        self.dataY = dataY
        self.dataName = dataName
        self.dataFeature = np.arange(self.dataX.shape[1])

        # 对数据01 归一化，将每一列的数据缩放到 [0,1]之间
        self.dataX = MinMaxScaler().fit_transform(self.dataX)
        # self.X_test = MinMaxScaler().fit_transform(self.X_test)
        print("进行filter，提取一部分数据")
        # filter operator
        filter = FilterOfData(dataX=self.dataX, dataY=self.dataY)
        a=time.time()
        #  compute relifF score
        self.scoreOfRelifF = filter.computerRelifFScore()
        b=time.time()-a
        print(b)
        # 进行filter过滤数据 ----  默认采用的是膝节点法
        self.scoreOfRelifF, self.dataX, self.dataFeature = filter.filter_relifF(
            scoreOfRelifF=self.scoreOfRelifF, dataX=self.dataX, dataFeature=self.dataFeature)
        # self.X_test = filter.filter_relifF_test(scoreOfRelifF=self.scoreOfRelifF, dataX=self.X_test, dataFeature=self.dataFeature)
        # 更新 特征长度
        self.dataFeatureNum = len(self.dataFeature)
        self.allDataIndex = np.arange(self.dataX.shape[1])
        self.globalFN = self.dataFeatureNum
        self.globalDI = np.arange(self.dataX.shape[1])
        self.dataX_global = self.dataX
        # 计算互信息
        self.globalinforgain = FilterOfData(dataX=self.dataX, dataY=self.dataY).computerInforGain()
        self.inforgain = self.globalinforgain

    # initialization parameters
    def setParameter(self,populationNum,iteratorTime,p,eta,c,mu_F,mu_CR,bm,cv):
        self.populationNum = populationNum
        self.iteratorTime = iteratorTime
        self.p = p
        self.eta = eta
        self.c = c
        self.mu_F = mu_F
        self.mu_CR = mu_CR
        self.bm = bm
        self.CV = cv

    # 染色体
    class Chromosome:
        # 每一个个体所选择出来的特征组合   --- 二进制
        featureOfSelect = np.asarray([])
        # 每一个个体所选择出来的特征组合   --- 实数制
        featureOfSelectNum = np.asarray([])
        # 特征组合的长度
        numberOfSolution = 0
        # 特征组合的索引
        indexOfSolution = np.asarray([])
        # 所选特征长度占总长度的比例
        proportionOfFeature = 0
        # 当前的选择的fitness
        mineFitness = 0
        # 所有数据的索引
        allDataIndex = np.asarray([])
        # 选择特征索引
        selectIndex = np.asarray([])
        bestFit = 1
        bestSolution = np.asarray([])
        bestSolutionNum = np.asarray([])

        # 该个体上抛弃的最优个体
        XBest_fit = 1
        XBest_solutionNum = np.asarray([])
        # 该个体上抛弃的次优个体
        XSecondBest_fit = 1
        XSecondBest_solutionNum = np.asarray([])

        # 每一个个体变异时的步长,    # 差分进化中变异操作的F 缩放因子
        F = 0
        # 每一个个体的交叉率
        CR = 0
        # 该个体的状态标记
        flag = 0
        # 停滞状态
        SC = 0

        def __init__(self, moea, mode, save):
            self.allDataIndex = moea.allDataIndex
            if mode == "init":
                self.initOfChromosome(MFDE=moea)
            elif mode == "iterator":
                self.iteratorOfChromosome(MFDE=moea)
            elif mode == "save":
                self.initOfSaveChromosome(MFDE=moea,SAVE=save)
            elif mode == "initsave":
                self.initSaveChromosome(MFDE=moea)

        # 自定义每一个个体的变异步长和交叉率
        def setFAndCR(self,mode):
            while  self.F <= 0 :
                # 产生该个体的F
                self.F = np.random.normal(mode.mu_F,0.1)

            if self.F > 1:
                self.F = 1

            while self.CR <= 0:
                # 产生该个体的CR
                #self.CR = np.random.standard_cauchy(mode.mu_CR,0.1)
                self.CR = cauchy.rvs(loc=mode.mu_CR, scale=0.1, size=1)[0]

            if self.CR > 1:
                self.CR = 1

        # 用于 中间产生新的个体时的初始化
        def iteratorOfChromosome(self, MFDE):
            self.featureOfSelect = np.zeros(MFDE.dataFeatureNum)
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()

        # 初始化
        def initOfChromosome(self, MFDE):
            self.featureOfSelect = np.zeros(MFDE.dataFeatureNum)
            self.featureOfSelectNum = np.zeros(MFDE.dataFeatureNum)
            # 随机提取 为 1 的索引
            randArray = np.asarray([])
            while len(randArray) < 2:
                randArray = sample(np.arange(MFDE.dataFeatureNum).tolist(), np.random.randint(MFDE.dataFeatureNum))
                # print(len(randArray))
            self.featureOfSelect[randArray] = 1
            for i in range(MFDE.dataFeatureNum):
                if self.featureOfSelect[i] == 1:
                    self.featureOfSelectNum[i] = np.random.uniform(MFDE.eta, 1)
                else:
                    self.featureOfSelectNum[i] = np.random.uniform(0, MFDE.eta)
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            # 计算acc
            feature_x = MFDE.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=MFDE.dataY, CV=MFDE.CV)

            if self.mineFitness < self.bestFit:
                self.bestFit = self.mineFitness
                self.bestSolution = self.featureOfSelect
                self.bestSolutionNum = self.featureOfSelectNum

        def initOfSaveChromosome(self, MFDE, SAVE):
            self.allDataIndex = MFDE.allDataIndex
            self.featureOfSelect = np.zeros(MFDE.dataFeatureNum)
            self.featureOfSelectNum = np.zeros(MFDE.dataFeatureNum)
            # 提取 为 1 的索引
            common_elements = np.intersect1d(MFDE.allDataIndex, SAVE.selectIndex)
            indices = np.where(np.isin(MFDE.allDataIndex, common_elements))[0]
            self.featureOfSelect[indices] = 1
            self.featureOfSelectNum[indices] = np.random.uniform(MFDE.eta, 1)
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            self.mineFitness = SAVE.mineFitness
            # 计算acc
            if self.mineFitness < self.bestFit:
                self.bestFit = self.mineFitness
                self.bestSolution = self.featureOfSelect
                self.bestSolutionNum = self.featureOfSelectNum
            return self

        def initSaveChromosome(self, MFDE):
            self.featureOfSelect = np.zeros(MFDE.dataFeatureNum)
            self.featureOfSelectNum = np.zeros(MFDE.dataFeatureNum)
            result =  np.array([rank for _, rank in sorted(zip(MFDE.sorted_feature, range(len(MFDE.sorted_feature))))])
            while np.all(self.featureOfSelect == 0):
                for i in result:
                    pop = 0.2 + 0.6 * (MFDE.dataFeatureNum - i - 1) / (MFDE.dataFeatureNum - 1)
                    if np.random.rand() < pop:  # (0.2~0.8)
                        self.featureOfSelect[i] = 1
                    else:
                        self.featureOfSelect[i] = 0
                    if self.featureOfSelect[i] == 1:
                        self.featureOfSelectNum[i] = np.random.uniform(MFDE.eta, 1)
                    else:
                        self.featureOfSelectNum[i] = np.random.uniform(0, MFDE.eta)
            # 自我保存，保存获取了多少的特征，为1特征的数量为多少个
            self.getNumberOfSolution()
            # 计算acc
            feature_x = MFDE.dataX[:, self.featureOfSelect == 1]
            self.mineFitness = 1 - fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=MFDE.dataY, CV=MFDE.CV)

            if self.mineFitness < self.bestFit:
                self.bestFit = self.mineFitness
                self.bestSolution = self.featureOfSelect
                self.bestSolutionNum = self.featureOfSelectNum

        # 得到选择特征数量是多少个
        def getNumberOfSolution(self):
            # 得到选择的组合里面 为1 的索引为多少个
            index = np.where(self.featureOfSelect == 1)[0]
            # 索引在dataFeature中所对应的值
            self.selectIndex = self.allDataIndex[index]
            # 存放每一个被选择的索引索引
            self.indexOfSolution = np.copy(index)
            # 得到的索引的长度是多少
            self.numberOfSolution = index.shape[0]
            # 获得比例
            self.proportionOfFeature = self.numberOfSolution / self.featureOfSelect.shape[0]

    # 初始化种群
    def initPopulation(self):
        savenum = len(self.saveArchive)
        for i in range(0, savenum):
            chromosome = self.Chromosome(moea=self, mode="save", save=self.saveArchive[i])
            self.population_DE = np.append(self.population_DE, chromosome)
        # 种群
        if savenum == 0:
            for i in range(0, int(self.populationNum)):
                chromosome = self.Chromosome(moea=self, mode="init", save=None)
                self.population_DE = np.append(self.population_DE, chromosome)
        else:
            for i in range(0, int(self.populationNum) - savenum):
                chromosome = self.Chromosome(moea=self, mode="initsave", save=None)
                self.population_DE = np.append(self.population_DE, chromosome)

    def sortFeature(self):
        feature_counts = {feature: 0 for feature in self.allDataIndex}
        for solution in self.saveArchive:
            selected_features = solution.selectIndex
            for feature in selected_features:
                if feature in feature_counts:
                    feature_counts[feature] += 1
        features = list(feature_counts.keys())
        counts = list(feature_counts.values())
        relifF_weights = self.scoreOfRelifF[self.allDataIndex]
        feature_df = pd.DataFrame({
            'Feature': features,
            'Count': counts,
            'ReliefF_Weight': relifF_weights
        })
        feature_array = feature_df.sort_values(by=['Count', 'ReliefF_Weight'], ascending=[False, False])
        self.sorted_feature = feature_array['Feature'].values.tolist()

    # 修改 mu_F
    def modify_F(self,set_F):
        #  Lehmer mean
        if len(set_F) == 0:
            mean_L = np.random.randn()
        else:
            mean_L = np.sum(set_F ** 2) / (np.sum(set_F) + 0.01)
        # 更新mu_F
        self.mu_F = (1 - self.c) * self.mu_F + self.c * mean_L

    # 和 mu_CR
    def modify_CR(self,set_CR):
        # arithmetic mean
        if len(set_CR) == 0:
            mean_A = np.random.randn()
        else:
            mean_A = np.mean(set_CR)
        # 更新mu_CR
        self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean_A

    # 得到当代优秀个体用于变异操作
    def getBestIndividualArray(self):
        # 用于存放每一个个体的 error
        errorArray = np.zeros(self.populationNum)
        # 进行迭代获取每一个个体的error
        for i in range(self.populationNum):
            errorArray[i] = self.population_DE[i].mineFitness

        # 进行排序
        errIndex = np.argsort(errorArray)

        # 去除最优解作为全局最优解
        if errorArray[errIndex[0]] < self.globalFitness:
            self.globalFitness = errorArray[errIndex[0]]
            self.globalSolution = self.population_DE[errIndex[0]].featureOfSelect

        # 输出最优的前三个个体
        for i in range(3):
            print("acc=",1 - errorArray[errIndex[i]],"  len=",self.population_DE[errIndex[i]].numberOfSolution)


        # 最优个体数组
        bestArray = self.population_DE[errIndex[:int(self.p * self.populationNum)]]
        bestIndex = errIndex[:int(self.p * self.populationNum)]
        return bestArray,bestIndex

    # 变异操作
    def mutateOperater(self,individual_i,X_best,X_r1,X_r2):
        # 获得该个体需要进行差分的solution
        X_i = individual_i.featureOfSelectNum
        vector_1 = X_best - X_i
        vector_2 = X_r1 - X_r2
        # 进行差分
        V_i = X_i + individual_i.F * vector_1 + individual_i.F * vector_2

        # 越界修复
        V_i = np.where(V_i < 0,(0 + X_i)/2,V_i)
        V_i = np.where(V_i > 1,(1 + X_i)/2,V_i)

        return V_i

    # 交叉算子
    def crossoverOperater(self,individual_i,V_i):
        U_i = np.zeros(self.dataFeatureNum)
        # 获得了该个体的CR
        CR = individual_i.CR
        # 选择一个 位置， 保证该位置可以进行 从V_i上获得数据
        indexOfCR = np.random.randint(self.dataFeatureNum)
        # 进行交叉
        for i in range(self.dataFeatureNum):
            # 获得一个随机数
            rand_1 = np.random.randn()
            if rand_1 <= CR or i == indexOfCR:
                U_i[i] = V_i[i]
            else:
                U_i[i] = individual_i.featureOfSelectNum[i]
        return U_i

    # 选择算子---origin
    def selectOperator(self,U_i,individual_i,set_F,set_CR):
        # 将实数制转化为二进制类型
        U_i_bin = np.where(U_i > self.eta,1,0)
        # 获得 所选择的特征数据
        feature_x = self.dataX[:, U_i_bin == 1]
        tmp_u_len = len(np.where(U_i_bin == 1)[0])
        if tmp_u_len > 0:
            # 计算fit
            error = 1 - fitnessFunction_KNN_CV(findData_x=feature_x,
                                            findData_y=self.dataY, CV=self.CV)
        else:
            error = 1
        # 跟新
        if error < individual_i.mineFitness:
            # 产生一个新的个体
            newIndevidual = self.Chromosome(moea=self, mode="iterator", save=None)
            # 这个新个体将遗传 原始个体的 F 和 CR
            newIndevidual.F = individual_i.F
            newIndevidual.CR = individual_i.CR
            # 将实数制类型的solution保存
            newIndevidual.featureOfSelectNum = U_i
            # 将二进制也保存起来
            newIndevidual.featureOfSelect = U_i_bin
            # 保存fitness
            newIndevidual.mineFitness = error
            # 更新
            newIndevidual.allDataIndex = individual_i.allDataIndex
            # 更新数据
            newIndevidual.getNumberOfSolution()
            # 将失败的父类保存到inferiorPop中
            self.inferiorPop = np.append(self.inferiorPop,individual_i)
            # 将成功跟新的 F 和 CR 保存
            set_F = np.append(set_F,individual_i.F)
            set_CR = np.append(set_CR,individual_i.CR)

            # 和最优解比较一下
            if newIndevidual.mineFitness < self.globalFitness:
                self.globalFitness = newIndevidual.mineFitness
                self.globalSolution = newIndevidual.featureOfSelect
                self.isRecursive = True
            elif newIndevidual.mineFitness == self.globalFitness and (
                    newIndevidual.numberOfSolution < len(np.where(self.globalSolution==1)[0])):
                self.globalSolution = newIndevidual.featureOfSelect
                self.isRecursive = True
            return newIndevidual,set_F,set_CR
        else:
            return individual_i,set_F,set_CR

    # 得到下一代
    def getNextPopulation(self):
        # 用于存放每一个成功存活的训练向量 的 F
        set_F = np.asarray([])
        # 用于存放每一个成功存活的训练向量 的 CR
        set_CR = np.asarray([])
        # 存放最优个体
        spArray = np.asarray([])
        # 获得前 p% 个的优秀个体
        bestArray,bestIndex = self.getBestIndividualArray()

        # 开始迭代
        for i in range(self.populationNum):
            # 获得第一个个体
            individual_i = self.population_DE[i]
            # 设置该个体的F和CR
            individual_i.setFAndCR(mode=self)
            # 随机选择最优个体从bestIndex选择，获得该个体在原始种群中的索引
            best_Index = i
            while best_Index == i :
                best_Index = sample(bestIndex.tolist(),1)[0]
            # 获得了第i个个体变异时的最优solution
            X_best = self.population_DE[best_Index].featureOfSelectNum
            # 获得随机个体
            randIndividualIndex = i
            # 如果选择到了 需要进行差分的个体 或者是差分个体，就需要进行重新选择
            while randIndividualIndex == i or randIndividualIndex == best_Index:
                randIndividualIndex = np.random.randint(0,self.populationNum)
            X_r1 = self.population_DE[randIndividualIndex].featureOfSelectNum
            # 将原始种群中 删除掉上述 找到的三个个体
            tempPop = np.delete(self.population_DE,[best_Index,randIndividualIndex,i])
            # tempPop种群和inferiorPop结合
            tempPop = np.concatenate([tempPop,self.inferiorPop])
            # 在这个新的种群中抽取一个个体作为最后一个变异所需的个体
            X_r2 = sample(tempPop.tolist(),1)[0].featureOfSelectNum

            # 变异
            V_i = self.mutateOperater(individual_i=individual_i,X_best=X_best
                                      ,X_r1=X_r1,X_r2=X_r2)
            # 交叉
            U_i = self.crossoverOperater(individual_i=individual_i,V_i=V_i)
            # 选择
            tmpInd, set_F, set_CR = self.selectOperator(U_i=U_i,
                        individual_i=individual_i,set_F=set_F,set_CR=set_CR)
            # 添加个体到子代种群
            spArray = np.append(spArray, tmpInd)
        # 合并
        conPop = np.concatenate((spArray, self.population_DE), axis=0)
        # 决策空间去重
        conPop = self.deleteDuolicate(conPop=conPop)
        # 非支配排序
        if conPop.shape[0] < self.populationNum:
            # 获得需要添加的个体数量
            addNum = self.populationNum - conPop.shape[0]
            for i in range(addNum):
                # 添加新个体
                conPop = np.append(conPop, self.Chromosome(moea=self, mode="init", save=None))
            self.population_DE = conPop

            pfArray = getPF(pop=conPop)

        else:
            pfArray, self.population_DE = nonDominatedSort_PFAndPop(
                parentPopulation=conPop, populationNum=self.populationNum)
        # 目标空间去重
        non_pf_pop = np.asarray([ind for ind in self.population_DE if ind not in pfArray])
        conPop = self.objectdeleteDuolicate(conPop=non_pf_pop,pfArray=pfArray)
        if conPop.shape[0] < self.populationNum:
            # 获得需要添加的个体数量
            addNum = self.populationNum - conPop.shape[0]
            for i in range(addNum):
                # 添加新个体
                conPop = np.append(conPop, self.Chromosome(moea=self, mode="init", save=None))
            self.population_DE = conPop
        # 将pfArray添加到 globalArchive
        conPop = np.concatenate((pfArray, self.localArchive), axis=0)
        # 去重
        conPop = self.deleteDuolicate(conPop=conPop)
        if len(conPop) > len(self.localArchive):
            # 非支配pf
            self.localArchive = getPF(conPop)

        # 更新 mu_F mu_CR
        self.modify_F(set_F=set_F)
        self.modify_CR(set_CR=set_CR)
        self.globalFitArray.append(self.localArchive[0].mineFitness)

    # 决策空间去重
    def deleteDuolicate(self, conPop):
        # 首先合并solution
        temp = np.asarray(conPop[0].featureOfSelect)
        for i in range(1,len(conPop)):
            temp = np.vstack((temp,conPop[i].featureOfSelect))
        # 转化为dataFrame
        df = pd.DataFrame(temp)
        # 去重
        df_dup = df.drop_duplicates()
        # 获得索引
        index_dup = df_dup.index.values
        # 提取相对应的个体
        new_pop = conPop[index_dup]
        return new_pop

    # 目标空间去重
    def objectdeleteDuolicate(self, conPop, pfArray):
        if len(conPop) < 2:
            return conPop  # 如果种群大小小于2，直接返回

        # 计算每个个体的适应度值（特征数量和误差）
        fitness_values = np.array([(ind.mineFitness, ind.numberOfSolution) for ind in conPop])
        # 找出具有相同适应度值的个体（重复解）
        unique_fitness, indices, counts = np.unique(fitness_values, return_index=True, return_counts=True, axis=0)
        duplicate_indices = np.where(counts > 1)[0]

        # 初始化新种群
        new_pop = np.asarray([])
        # 遍历每组重复解
        for idx in duplicate_indices:
            # 获取重复解的索引
            dup_group = np.where((fitness_values == unique_fitness[idx]).all(axis=1))[0]
            # 初始化加权后的汉明距离数组
            if len(dup_group) == 2:  # 如果组里只有两个重复解
                # 计算特征频率（非支配解中的特征频率）
                pf_feature_frequency = np.sum([ind.featureOfSelect for ind in pfArray], axis=0) / len(pfArray)
                # 提取重复解的特征选择矩阵
                feature_matrix = np.array([ind.featureOfSelect for ind in conPop[dup_group]])
                # 计算综合评分：特征选择 × 频率 × 互信息
                combined_scores = np.sum(feature_matrix * pf_feature_frequency * self.inforgain, axis=1)
            else:
                weighted_hamming = np.zeros(len(dup_group))
                # 遍历每个个体的特征选择情况
                for i in range(len(dup_group)):
                    # 获取当前个体所选择的特征索引
                    for j in range(len(dup_group)):
                        if i != j:
                            # 计算加权汉明距离
                            diff_features = (conPop[dup_group[i]].featureOfSelect != conPop[dup_group[j]].featureOfSelect)
                            weighted_hamming[i] += np.sum(self.inforgain * diff_features)
                            combined_scores = weighted_hamming
                # 如果存在相同值，计算特征频率和互信息
                if len(np.unique(weighted_hamming)) < len(weighted_hamming):
                     # 计算特征频率（非支配解中的特征频率）
                     pf_feature_frequency = np.sum([ind.featureOfSelect for ind in pfArray], axis=0) / len(pfArray)
                     # 提取重复解的特征选择矩阵
                     feature_matrix = np.array([ind.featureOfSelect for ind in conPop[dup_group]])
                     # 计算综合评分：特征选择 × 频率 × 互信息
                     combined_scores = np.sum(feature_matrix * pf_feature_frequency * self.inforgain, axis=1)

            # 选择评分最高的解
            best_idx = dup_group[np.argmax(combined_scores)]
            new_pop = np.append(new_pop, conPop[best_idx])

        # 合并不重复的解和选择后的重复解
        if len(duplicate_indices) !=0:
            unique_indices = np.setdiff1d(np.arange(len(conPop)), np.concatenate(
                [np.where((fitness_values == uf).all(axis=1))[0] for uf in unique_fitness[duplicate_indices]]))
            new_pop = np.concatenate([new_pop, conPop[unique_indices], pfArray])
            return np.array(new_pop)
        conPop = np.concatenate([conPop, pfArray])
        return conPop


    # 提取前bm个 个体，并且排序
    def getSubData(self):
        # 判断局部档案和全局档案之间的关系
        # 判断pfArray中个体的长度是否等于 self.globalArchive中个体的长度
        self.saveArchive = asarray([])#------------------
        for i in range(len(self.localArchive)):
            self.localArchive[i].featureOfSelect = np.zeros(self.globalFN)
            self.localArchive[i].featureOfSelect[self.localArchive[i].selectIndex] = 1
            self.localArchive[i].allDataIndex = self.globalDI
            self.localArchive[i].getNumberOfSolution()
            self.saveArchive = np.append(self.saveArchive,self.localArchive[i])  # 下一次递归的初始个体---------------
        gl_num = self.globalArchive.shape[0]
        self.globalArchive = np.concatenate((self.globalArchive, np.asarray(self.localArchive)), axis=0)
        if len(self.globalArchive) == 1:
            pass
        else:
            # 去重
            self.globalArchive = self.deleteDuolicate(conPop=self.globalArchive)
        if self.globalArchive.shape[0] > gl_num and self.globalArchive.shape[0] != 1:
            self.globalArchive = getPF(pop=self.globalArchive)
            self.globalArchive = np.asarray(self.globalArchive)
        # 组合数据子集
        conArray = [[]]
        conArray[0] = np.where(self.globalArchive[0].featureOfSelect == 1)[0].tolist()
        self.bm = len(self.globalArchive)
        for i in range(1, self.bm):
            conArray.append(np.where(self.globalArchive[i].featureOfSelect == 1)[0].tolist())
        # 添加一个条件数组，当为false时则该位置所对应的的特征子集被抛弃
        isSave = np.ones(self.bm)
        # 被用于下一轮递归的数据集的特征索引
        subDataIndex = conArray[0]
        # 将该数据集中所选择的特征提取出来
        for i in range(self.bm - 1, -1, -1):
            if i == 0:
                break
            len_i = len(conArray[i])
            for j in range(i - 1, -1, -1):
                len_j = len(conArray[j])
                interArray = np.intersect1d(conArray[i], conArray[j])
                len_c_i = len(interArray)
                if len_c_i == len_i and len_c_i < len_j:
                    # 删除 第 i 个
                    isSave[i] = 0
                    break
        # 将isSave中为1的特征子集全部保存下来
        for i in range(1, self.bm):
            if isSave[i]:
                subDataIndex = np.union1d(subDataIndex, conArray[i])
        # 将这些特征都置为1
        self.dataX = self.dataX_global[:,subDataIndex]
        self.allDataIndex = self.globalDI[subDataIndex]
        self.inforgain = self.globalinforgain[subDataIndex]

    # 将每一轮的pareto写入到excel

    # def print_excel(self, index, path_xlsx):
    #     if not os.path.exists(path_xlsx):
    #         # 创建新工作簿
    #         wb = openpyxl.Workbook()
    #         # 获取默认工作表
    #         sheet = wb.active
    #         sheet.title = "gloalPF6"  # 设置工作表名称
    #         # 保存文件
    #         wb.save(filename=path_xlsx)
    #     wb = openpyxl.load_workbook(filename=path_xlsx)
    #     global_pf = wb["gloalPF6"] # 2 3
    #     global_pf.cell(row=1, column=1 + index, value="acc")
    #     global_pf.cell(row=1, column=2 + index, value="len")
    #     global_pf.cell(row=1, column=3 + index, value="proportion")
    #     global_pf.cell(row=1, column=4 + index, value="selectIndex")
    #     for i in range(len(self.localArchive)):
    #         # 添加acc到xlsx
    #         global_pf.cell(row=2 + i, column=1 + index, value = self.localArchive[i].mineFitness)
    #         # 添加len到xlsx
    #         tmp_len = len(np.where(self.localArchive[i].featureOfSelect == 1)[0])
    #         global_pf.cell(row=2 + i, column=2 + index, value = tmp_len)
    #         global_pf.cell(row=2 + i, column=3 + index, value = tmp_len / self.globalFN)
    #         global_pf.cell(row=2 + i, column=4 + index, value = str(self.localArchive[i].selectIndex.tolist()))
    #         wb.save(filename=path_xlsx)

    # 运行
    def run(self, const):
        print("第", const, "次递归")
        # 初始化函数
        self.initPopulation()
        # 进行运行
        runTime = 0
        while runTime < self.iteratorTime:
            print("第", runTime + 1, "代")
            self.getNextPopulation()
            runTime += 1

        index = (const - 1) * 4
        path_xlsx = "C:/Users/Dec/Desktop/实验/" + "IGD" + ".xlsx"
        #path_xlsx = "F:/MachineLearningBackUp/RecursiveCompare/RecursiveDE/" + self.dataName + ".xlsx"
        # self.print_excel(index = index, path_xlsx = path_xlsx)


        if self.isRecursive:
            # 提取前bm个
            self.getSubData()
            self.sortFeature()
            self.localArchive = np.asarray([])
            self.dataFeature = np.arange(self.dataX.shape[1])
            self.dataFeatureNum = self.dataX.shape[1]
            self.isRecursive = False
            self.population_DE = np.asarray([])
            self.inferiorPop = np.asarray([])
            self.mu_F = 0.5
            self.mu_CR = 0.5
            const += 1
            self.run(const=const)
        else:
            print("best acc = ", 1 - self.globalFitness)
            print("best len = ", len(np.where(self.globalSolution == 1)[0]))
            fitness.append(1 - self.globalFitness)
            feature.append(len(np.where(self.globalSolution == 1)[0]))
            # 得到最终前沿
            if not os.path.exists(path_xlsx):
                # 如果文件不存在，创建新工作簿
                wb = openpyxl.Workbook()
                # 删除默认创建的 Sheet（如果有）
                if "Sheet" in wb.sheetnames:
                    wb.remove(wb["Sheet"])
                # 创建新的工作表
                ws = wb.create_sheet(ducName)
                wb.save(filename=path_xlsx)
            else:
                # 如果文件存在，加载工作簿
                wb = openpyxl.load_workbook(filename=path_xlsx)
                # 检查工作表是否存在
                if ducName in wb.sheetnames:
                    ws = wb[ducName]
                else:
                    ws = wb.create_sheet(ducName)

            # 找到当前工作表的最后一列
            max_col = ws.max_column

            # 如果工作表是新建的，max_col 可能是 1（空表）
            # 如果工作表已有数据，max_col 会是最后一列

            # 确定写入的起始列（避免覆盖已有数据）
            start_col = max_col + 1 if max_col > 1 else 1

            # 写入数据
            for i in range(len(self.globalArchive)):
                print(1 - self.globalArchive[i].mineFitness, " ", self.globalArchive[i].numberOfSolution)
                ws.cell(row=1 + i, column=start_col, value=1 - self.globalArchive[i].mineFitness)
                ws.cell(row=1 + i, column=start_col + 1, value=self.globalArchive[i].numberOfSolution)

            # 保存文件
            wb.save(filename=path_xlsx)
            print()
            print("迭代次数：", const)
            print("iteratorTime = ", self.iteratorTime)
            print("populationNum = ", self.populationNum)
            # feature_x1 = self.dataX_global[:, self.globalArchive[0].featureOfSelect == 1]
            # feature_x = self.X_test[:, self.globalArchive[0].featureOfSelect == 1]
            # Fitness = fitnessFunction_KNN_CV(findData_x=feature_x1, findData_y=self.dataY, CV=10)
            # mineFitness = fitnessFunction_KNN_CV(findData_x=feature_x, findData_y=self.Y_test, CV=10)
            # print("训练集acc = ", Fitness)
            # print("测试集acc = ", mineFitness)



if __name__ == '__main__':
    import time
    fitness = []
    feature = []
    times = []
    # ducName = ["arcene","Brain_Tumor_2","BreastCancer1","BreastCancer2","CLLSUB","CrohnDisease","Eleven_Tumor", "GLI", "Leukemia","Leukemia_3", "LungCancer", "Ovarian","Prostate", "SMKCAN187"]
    ducName = ["CLLSUB", "CrohnDisease", "Eleven_Tumor",
               "GLI", "Leukemia", "Leukemia_3", "LungCancer", "Ovarian", "Prostate", "SMKCAN187"]
    for ducName in ducName:
        for i in range(10):
            print(f"第{i}次循环")
            start = time.time()
            # ducName = "arcene"
            # ducName = "LungCancer"

            #ducName = "Brain_Tumor_2"
            # ducName = "Ovarian"
            path = "dataCSV/dataCSV_high/" + ducName + ".csv"
            # path = "/home/fanfan/dataCSV/dataCSV_high/" + ducName + ".csv"
            dataCsv = ReadCSV(path=path)
            print("获取文件数据")
            dataCsv.getData()

            genetic = Genetic(dataX=dataCsv.dataX, dataY=dataCsv.dataY, dataName=ducName)
            genetic.setParameter(populationNum=100, iteratorTime=50, mu_F=0.5,
                                 mu_CR=0.5, eta=0.6, c=0.1, p=0.05, bm=5, cv=5)
            const = 1
            genetic.run(const=const)
            # df = pd.DataFrame(genetic.globalFitArray, columns=['Numbers'])
            # # 写入 Excel 文件
            # file_path = "C:/Users/Dec/Desktop/实验/"+ducName+"-output.xlsx"
            # df.to_excel(file_path, index=False)  # index=False 表示不写入行索引
            #
            # print(f"all time = {time.time() - start} seconds")
            # times.append(time.time() - start)
            # try:
            #     wb = openpyxl.load_workbook('C:/Users/Dec/Desktop/实验/'+ducName+'-30.xlsx')
            #     ws = wb.active
            # except FileNotFoundError:
            #     # 如果文件不存在，则创建一个新的工作簿
            #     wb = openpyxl.Workbook()
            #     ws = wb.active
            # ws.cell(row=1, column=i + 1, value=fitness[i])
            # ws.cell(row=2, column=i + 1, value=feature[i])
            # ws.cell(row=3, column=i + 1, value=times[i])
            # wb.save('C:/Users/Dec/Desktop/实验/'+ducName+'-30.xlsx')




