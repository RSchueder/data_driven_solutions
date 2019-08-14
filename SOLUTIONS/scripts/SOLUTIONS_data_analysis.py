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
###############################################################################
# CLASSIFIED PAIRPLOT
###############################################################################
classified = pd.read_excel('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\ResultsCaseStudies_EoE_FoE.xlsx',sheetname = 'classified')
sns.pairplot(classified,hue = 'class',palette = 'hls')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\EoE_impact_manualPAFclassified_pairplot.png',dpi = 200)

###############################################################################
# UNCLASSIFIED PAIRPLOT
############################################################################### 
unclassified = pd.read_excel('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\ResultsCaseStudies_EoE_FoE.xlsx',sheetname = 'unclassified')
unclassified.drop(['PAF'],axis = 1, inplace = True)
unclassified.loc[unclassified['%Export'] > 1,'%Export']  = np.nan
unclassified.dropna(axis =0, inplace = True)
unclassified.loc[(unclassified['logKOW0'] > 8) & (unclassified['EoE'] < -8),'type'] = '^KOW'
# unclassified['%Fugacity'] = np.log10(unclassified['%Fugacity'])
sns.pairplot(unclassified,hue = 'type',palette = 'hls')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\EoE_type_autoclassified_pairplot.png',dpi = 200)

###############################################################################
# REGRESSION SETS AND PAIRPLOT
###############################################################################
regression = pd.read_excel('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\ResultsCaseStudies_EoE_FoE.xlsx',sheetname = 'EoE_FoE')

# DROP REDUNDANT PARAMETERS
regression.drop(['case','MW','E2SNO (kg/y)','Etot (kg/y)','E2SW (kmol/y)','FromSGW (kmol/y)','Stor','E/PNEC','EoE','PAF','logPAF'], inplace = True, axis = 1)
# REMOVE THOSE WITH POOR BALANCE
regression.loc[regression['Export'] > 1,'Export']  = np.nan
regression.dropna(axis = 0, inplace = True)
regression['E2S1 (kg/y)'] = np.log10(regression['E2S1 (kg/y)'])
regression['E2RIVTOT (kg/y)'] = np.log10(regression['E2RIVTOT (kg/y)'])
regression['%Fugacity'] = np.log10(regression['%Fugacity'])

regression.loc[(regression['logKOW0'] > 8) & (regression['logEoE'] < -8),'type'] = 'KOW0'

# FOR TYPE PAIRPLOT FULL SET
regpair = regression.copy()

# FOR MULTIVARIATE REGRESSION, DROP ENDOGENOUS LATER
regmulti = regression.copy()

###############################################################################
# CREATE REGRESSION SET FOR ALL SUBSTANCES
###############################################################################

# REMOVE CERTAIN VARIABLES
# regmulti IS REGRESSION SET, WITH SELECT REGRESSION VARIABLES
regmulti = sm.add_constant(regmulti)
regmulti.dropna(axis = 0, inplace = True)
regmultiX1 = regmulti.copy()
EoE = regmulti['logEoE']

# this should be doable in 2 lines, no?
regmultiX2 = regmulti.copy()
regmultiX2.loc[regmultiX2['FoE'] < 1e-20] = np.nan
regmultiX2.dropna(axis = 0, inplace = True)
FoE = regmultiX2['FoE']

# CREATE SUBSTANCES SUIBSETS
pharmaX1 = regmulti.loc[regmulti['type'] == '1_Pharma'].copy()
pharmaEoE = pharmaX1['logEoE']

pharmaX2 = pharmaX1.copy()
pharmaX2.loc[pharmaX2['FoE'] < 1e-20] = np.nan
pharmaX2.dropna(axis = 0, inplace = True)
pharmaFoE = pharmaX2['FoE']

reachX1 = regmulti.loc[regmulti['type'] == '3_REACH'].copy() 
reachEoE = reachX1['logEoE']

reachX2 = reachX1.copy()
reachX2.loc[reachX2['FoE'] < 1e-20] = np.nan
reachX2.dropna(axis = 0, inplace = True)
reachFoE = reachX2['FoE']

pestX1 = regmulti.loc[regmulti['type'] == '2_Pest'].copy()
pestEoE = pestX1['logEoE']

pestX2 = pestX1.copy()
pestX2.loc[pestX2['FoE'] < 1e-20] = np.nan
pestX2.dropna(axis = 0, inplace = True)
pestFoE = pestX2['FoE']

# AFTER CHARACTERIZATION, DROP LABELS and ENDOGENOUS VARIABLE AS THEY CANNOT BE 
# INCLUDED IN REGRESSION 
# EOE SET
regmultiX1.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
pharmaX1.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
pestX1.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
reachX1.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
# FOE SET
regmultiX2.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
pharmaX2.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
pestX2.drop(['type','logEoE','FoE'], inplace = True, axis = 1)
reachX2.drop(['type','logEoE','FoE'], inplace = True, axis = 1)

###############################################################################
# CREATE REGRESSION MODELS - EOE
###############################################################################

regEoEmodel = sm.OLS(EoE, regmultiX1.astype(float)).fit()
pharmaEoEmodel = sm.OLS(pharmaEoE, pharmaX1.astype(float)).fit()
pestEoEmodel = sm.OLS(pestEoE, pestX1.astype(float)).fit()
reachEoEmodel = sm.OLS(reachEoE, reachX1.astype(float)).fit()

###############################################################################
# CREATE REGRESSION MODELS - FOE (CLEANED EARLIER IN AN ADDITIONAL STEP)
###############################################################################

regFoEmodel = sm.OLS(FoE, regmultiX2.drop(['const'], axis = 1).astype(float)).fit()
pharmaFoEmodel = sm.OLS(pharmaFoE, pharmaX2.drop(['const'], axis = 1).astype(float)).fit()
pestFoEmodel = sm.OLS(pestFoE, pestX2.drop(['const'], axis = 1).astype(float)).fit()
reachFoEmodel = sm.OLS(reachFoE, reachX2.drop(['const'], axis = 1).astype(float)).fit()

###############################################################################
# EXAMINE REGRESSION MODELS - EOE
###############################################################################
fig1, axes = plt.subplots(nrows = 2,ncols = 2, figsize = (8.5,11))
axes[0][0].plot(EoE,regEoEmodel.predict(regmultiX1),'o')
axes[0][0].set_xlabel('observed logEoE')
axes[0][0].set_ylabel('predicted logEoE')
axes[0][0].plot([-8,8],[-8,8],'--')

axes[0][1].plot(pharmaEoE,pharmaEoEmodel.predict(pharmaX1),'o')
axes[0][1].set_xlabel('observed logEoE PHARMA')
axes[0][1].set_ylabel('predicted logEoE PHARMA')
axes[0][1].plot([-8,8],[-8,8],'--')

axes[1][0].plot(pestEoE,pestEoEmodel.predict(pestX1),'o')
axes[1][0].set_xlabel('observed logEoE PEST')
axes[1][0].set_ylabel('predicted logEoE PEST')
axes[1][0].plot([-8,8],[-8,8],'--')

axes[1][1].plot(reachEoE,reachEoEmodel.predict(reachX1),'o')
axes[1][1].set_xlabel('observed logEoE REACH')
axes[1][1].set_ylabel('predicted logEoE REACH')
axes[1][1].plot([-8,8],[-8,8],'--')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\EoE_multivariate_regression.png',dpi = 200)

###############################################################################
# EXAMINE REGRESSION MODELS - FOE
###############################################################################

fig2, axes = plt.subplots(nrows = 2,ncols = 2, figsize = (8.5,11))
axes[0][0].plot(FoE,regFoEmodel.predict(regmultiX2.drop(['const'], axis = 1)),'o')
axes[0][0].set_xlabel('observed FoE')
axes[0][0].set_ylabel('predicted FoE')
axes[0][0].plot([0,1],[0,1],'--')

axes[0][1].plot(pharmaFoE,pharmaFoEmodel.predict(pharmaX2.drop(['const'], axis = 1)),'o')
axes[0][1].set_xlabel('observed FoE PHARMA')
axes[0][1].set_ylabel('predicted FoE PHARMA')
axes[0][1].plot([0,1],[0,1],'--')

axes[1][0].plot(pestFoE,pestFoEmodel.predict(pestX2.drop(['const'], axis = 1)),'o')
axes[1][0].set_xlabel('observed FoE PEST')
axes[1][0].set_ylabel('predicted FoE PEST')
axes[1][0].plot([0,1],[0,1],'--')

axes[1][1].plot(reachFoE,reachFoEmodel.predict(reachX2.drop(['const'], axis = 1)),'o')
axes[1][1].set_xlabel('observed FoE REACH')
axes[1][1].set_ylabel('predicted FoE REACH')
axes[1][1].plot([0,1],[0,1],'--')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\FoE_multivariate_regression.png',dpi = 200)

###############################################################################
# SINGLE VARIABLE REGRESSIONS
###############################################################################

par = ['logE/PNEC','logERIV/PNEC','%Fugacity','logKOW0','kdw']
for ii,pp in enumerate(par):
    fig = plt.figure(ii+20)
    ax = fig.add_axes([.1,0.1,0.9,0.9])
    X = regression[pp]
    X = sm.add_constant(X)
    Y = regression['logEoE']
    linmodel = sm.OLS(endog = Y, exog = X.astype(float)).fit()
    linpred = linmodel.predict(X)
    
    ax.plot(regression[pp],regression['logEoE'],'o')    
    ax.plot(X[pp],linpred,'o')
    
    ax.set_xlabel(pp)
    ax.set_ylabel('log EoE')

    pylab.savefig(('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\EoE_%s_regression.png')% pp.replace('/',''),dpi = 200)

###############################################################################
# EXPLORATORY FULL PAIR PLOTS 
###############################################################################

sns.pairplot(regpair,hue = 'type',palette = 'hls')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\EoE_type_autoclassified_fullregpar_pairplot.png',dpi = 200)

###############################################################################
# SMALLER MULTIVARIATE REGRESSIONS
###############################################################################

X = unclassified.drop(['EoE', 'type', 'CAS', 'FoE'], axis = 1).copy()
# X = sm.add_constant(X)
y = unclassified['EoE']
Emodel = sm.OLS(y, X.astype(float)).fit()
fig = plt.figure(80)
ax = fig.add_axes([.1,0.1,0.9,0.9])
ax.plot(y, Emodel.predict(X), 'o')
ax.set_xlabel('observed logEoE')
ax.set_ylabel('predicted logEoE')
ax.plot([-8,8], [-8,8], '--')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\EoE_small_set_regression.png', dpi = 200)

unclassified.loc[(unclassified['%Fugacity'] >= np.log10(1)),'type'] = 'H'
unclassified.loc[(unclassified['%Fugacity'] < np.log10(1)),'type'] = 'L'

subpair = unclassified[['EoE','ERIV/PNEC','%Fugacity']]
sns.pairplot(unclassified,hue = '%Fugacity',palette = 'hls')
pylab.savefig('d:\schueder\Documents\projects\SOLUTIONS\QP_ReferenceAnalysis\\FUGSHADE.png',dpi = 200)
