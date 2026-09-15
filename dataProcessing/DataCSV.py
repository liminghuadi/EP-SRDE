import os
import pandas as pd
import  numpy as np

# CSV的
class DataCSV:
    ducPath = ""
    #  除了class外的所有数据   ---->  用于计算relifF
    dataX = np.asarray([])
    #
    dataY = np.asarray([])

    def __init__(self,path):
        self.ducPath = path



    def getData(self):
        file_path = self.ducPath
        data = pd.read_csv(file_path)          # 获得数组
        self.dataX = data.values[:,0:-1]
        self.dataY = data.values[:,-1]


if __name__ == "__main__":
    dataCsv = DataCSV("D:\MachineLearningBackUp\\dataCSV\\Breast.csv")
    #dataCsv = DataCSV("D:\MachineLearningBackUp\data\ionosphere.csv")
    dataCsv.getData()

