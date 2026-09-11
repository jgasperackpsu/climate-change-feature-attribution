"""
reference:Das S, Suganthan P N. Differential evolution: A survey of the state-of-the-art[J]. IEEE transactions on evolutionary computation, 2010, 15(1): 4-31.
reference code link:https://github.com/guofei9987/scikit-opt
"""
from __future__ import division
from numpy import *
import random as rd

NP = 1000
size = 4
xMin = 0.001
xMax = 1
F = 0.5
CR = 0.8

pred1 = array([])
pred2 = array([])
pred3 = array([])
pred4 = array([])
y_true = array([])

y_pred = vstack((pred1, pred2, pred3, pred4)).T

XTemp = random.uniform(xMin, xMax, (NP, size))
XTemp = XTemp / XTemp.sum(axis=1, keepdims=True)

def calFitness(weights):
    ensemble_pred = y_pred.dot(weights)
    mape = mean(abs((y_true - ensemble_pred) / y_true)) * 100
    return mape

def mutation(XTemp, F):
    m, n = shape(XTemp)
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1, r2, r3 = rd.sample(range(m), 3)
        while i in (r1, r2, r3):
            r1, r2, r3 = rd.sample(range(m), 3)
        XMutationTmp[i] = XTemp[r1] + F * (XTemp[r2] - XTemp[r3])
        XMutationTmp[i] = clip(XMutationTmp[i], xMin, xMax)
    return XMutationTmp

def crossover(XTemp, XMutationTmp, CR):
    m, n = shape(XTemp)
    XCorssOverTmp = zeros((m, n))
    for i in range(m):
        for j in range(n):
            r = rd.random()
            XCorssOverTmp[i, j] = XMutationTmp[i, j] if r <= CR else XTemp[i, j]
    return XCorssOverTmp

def selection(XTemp, XCorssOverTmp, fitnessVal):
    m, n = shape(XTemp)
    for i in range(m):
        new_fitness = calFitness(XCorssOverTmp[i])
        if new_fitness < fitnessVal[i]:
            XTemp[i] = XCorssOverTmp[i]
            fitnessVal[i] = new_fitness
    return XTemp, fitnessVal

fitnessVal = array([calFitness(XTemp[i]) for i in range(NP)])

gen = 0
while gen <= 200:
    XMutationTmp = mutation(XTemp, F)
    XCorssOverTmp = crossover(XTemp, XMutationTmp, CR)
    XTemp, fitnessVal = selection(XTemp, XCorssOverTmp, fitnessVal)
    gen += 1

best_idx = argmin(fitnessVal)
print(f"Best Weights: {XTemp[best_idx]}")
