# -*- coding: utf-8 -*-
'''
Created on Jan 25, 2022
@author: Xingchen Li
'''

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random


# Auxiliary function1
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# Auxiliary function2
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# Implementation of SMO algorithm
def smoSimple(dataMat, classLabels, C, toler, maxIter):
    '''
    @dataMat    ：Data list
    @classLabels：Lab list
    @C          ：Weighting factor（The relaxation factor is added and the penalty term is introduced into the objective optimization function）
    @toler      ：Fault tolerance rate
    @maxIter    ：Maximum iteration
    '''
    # Convert list form to matrix or vector form
    dataMatrix = mat(dataMat)
    labelMat = mat(classLabels).transpose()
    # Initialize b=0 to get the matrix rows and columns
    b = 0
    m, n = shape(dataMatrix)
    # Create a vector with m rows and 1 column
    alphas = mat(zeros((m, 1)))
    # iterations time is 0
    iters = 0
    while(iters < maxIter):
        # Modify logarithmic of alpha
        alphaPairsChanged = 0
        # Iterate over samples in the sample set
        for i in range(m):
            # Calculate the predicted value of the support vector machine algorithm
            fXi = float(multiply(alphas, labelMat).T *
                        (dataMatrix*dataMatrix[i, :].T))+b
            # Calculate the error between the predicted value and the actual value
            Ei = fXi-float(labelMat[i])
            # If the KKT condition is not satisfied, that islabelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            # and alpha<C Or labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if(((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or
               ((labelMat[i]*Ei > toler) and (alphas[i] > 0))):
                # The second variable, alphaj, is selected at random
                j = selectJrand(i, m)
                # Calculate the predicted value of the data for the second variable

                fXj = float(multiply(alphas, labelMat).T*(dataMatrix *
                                                          dataMatrix[j, :].T)) + b
                # Calculate and test the difference from the actual value
                Ej = fXj - float(labelMat[j])
                # Record the original values of alphai and Alphaj for subsequent comparison
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # How can two alpha samples have different labels
                if(labelMat[i] != labelMat[j]):
                    # Find the corresponding upper and lower boundaries
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    # print("L==H")
                    continue
                # Calculate unclipped ALPHAJ according to the formula
                # ------------------------------------------
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T -\
                    dataMatrix[i, :]*dataMatrix[i, :].T -\
                    dataMatrix[j, :]*dataMatrix[j, :].T
                # If eta>=0, exit the loop
                if eta >= 0:
                    # print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # ------------------------------------------
                # If the changed alphaJ value does not change much, it will jump out of the loop
                if(abs(alphas[j]-alphaJold) < 0.00001):
                    # print("j not moving enough")
                    continue
                # Otherwise, calculate the corresponding ALphai value
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                # Then calculate the b value of phi for the two alpha cases
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]\
                    * dataMatrix[i, :].T - labelMat[j]*(alphas[j]-alphaJold) *\
                    dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold) *\
                    dataMatrix[i, :]*dataMatrix[j, :].T -\
                    labelMat[j]*(alphas[j]-alphaJold) *\
                    dataMatrix[j, :]*dataMatrix[j, :].T
                # if 0<alphai<C, then b=b1
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                # Otherwise if 0<alphaj=C, then b=b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                # Otherwise, alphai, alphaj=0 or C
                else:
                    b = (b1+b2)/2.0
                # If you go this far, the surface changes a pair of alpha values
                alphaPairsChanged += 1
                print("iters: %d i:%d,paird changed %d" % (iters, i, alphaPairsChanged))
        # Finally, determine whether there are alpha pairs that have changed, and proceed to the next iteration if not
        if(alphaPairsChanged == 0):
            iters += 1
        # Otherwise, set the number of iterations to 0 and continue the loop
        else:
            iters = 0
        print("iteration number: %d" % iters)
    # Returns the final b value and alpha vector
    return b, alphas


def calcWs(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(
        alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


def showClassifer(dataMat, labelMat, alphas, w, b):
    data_plus = []
    data_minus = []
    #
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(
        data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(
        data_minus_np)[1], s=30, alpha=0.7)
    #
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #
    for i, alpha in enumerate(alphas):
        if 0.6 > abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                        linewidth=1.5, edgecolor='red')
        if 0.6 == abs(alpha):
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                        linewidth=1.5, edgecolor='yellow')
    plt.show()