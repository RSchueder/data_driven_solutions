# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 05:23:01 2017

@author: schueder
"""

# WFLOW to dis 

# this script takes the xlsx output from W-FLOW and converts it to a delft3d
# discharge file

import pandas as pd
import numpy as np
import datetime as dt

def zadd(varagin):
    if len(str(varagin)) == 1:
        dat = '0' + str(varagin)
        return str(dat)
    else:
        return varagin
xls = pd.ExcelFile(r'data\LagunaLake\hydrology\Meteo and Streamflow Data\Wflow2Excel-2017-model_2004-2016.xlsx')
wflowf = xls.parse('Wflow-results', skiprows = 3)

wrefDate = dt.datetime.toordinal(dt.date(2004,1,1))

timestep = 1
rivSal = 0.017
rivTemp = 20.0

# assign dictionary of catch name and catch number

nums = wflowf['catch.nr']
nameDict = {}
for ii,nn in enumerate(nums):
    if nn in nameDict.keys():
        pass
    else:
        nameDict[(('%s') % nn)] = wflowf['catch.name'][ii]

disTick = 0
ref = pd.Timestamp(dt.date.fromordinal(wrefDate).year, dt.date.fromordinal(wrefDate).month,dt.date.fromordinal(wrefDate).day)
for ii,cc in enumerate(wflowf.columns):
    if 'SB' in str(cc):

        scName = nameDict[(('%s') % cc)]
    
        with open((r'data\LagunaLake\hydrology\Meteo and Streamflow Data\%s_WFlow.csv') % scName,'w') as disFile:
                # one of the subbasins
                disFile.write('date,%s\n' % scName)
                disTick = disTick + 1        
                tmpDf = wflowf[['time step', cc]].copy()
                tmpDf.dropna(inplace = True)
                dat = [ref + tmpDf['time step'].apply(lambda x: pd.Timedelta(days = x-731)), np.array(tmpDf[cc])]
                for ii,tt in enumerate(dat[0]):
                    disFile.write(str(tt) + ",")
                    disFile.write(str(dat[1][ii]))
                    #disFile.write(str(dat[1][ii]) + "    " + str(rivSal) + "    " + str(rivTemp))
                    disFile.write('\n')
                disFile.write('\n')


        