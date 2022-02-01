'''
Created on Jan 25, 2022
@author: Xingchen Li
'''

from kernel import *
from SMO_simp import *

if __name__ == "__main__":
    # dataArr, labelArr = loadDataSet("testSet.txt")
    # b, alphas = smoSimple(dataArr, labelArr, 0.01, 0.001, 40)
    # w = calcWs(dataArr, labelArr, alphas)
    # showClassifer(dataArr, labelArr, alphas, w, b)
    testRbf(k1=10000)