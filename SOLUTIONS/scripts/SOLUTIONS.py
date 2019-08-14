# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:40:56 2018

this is some code used to analyse contributors to toxicity in the streameu
model
this code is exploratory, and was not planned. As a result, it consists mostly
of explicit statements and is very long and verbose

regression = the seed dataset, minimal entry removal, only missing descriptor
data 
regpair = the full data set with only redundant data removal. for use in pairplots
regmulti = used to create multilinear regressions, used to create subsets for substance types

question, can I assigne X = df.drop['A'].copy() without affecting df?
this type of problem contributes to this script's length

@author: schueder
"""


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import statsmodels.api as sm
from sklearn import linear_model
clf = linear_model.LinearRegression()
plt.close()

plot_kws = {'s' : 20}
file = './data/ResultsCaseStudies_EoE_FoE.xlsx'

classified = pd.read_excel(file,sheetname = 'classified')
unclassified = pd.read_excel(file,sheetname = 'unclassified')
regression = pd.read_excel(file,sheetname = 'EoE_FoE')
