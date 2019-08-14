# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:43:39 2018

@author: schueder
"""

# prepare dat for AI

import numpy as np
import pandas as pd
import seaborn as sns
import keras

def make_ts(var):
    return pd.Timestamp(var)

process = True

if process:   
    excelFile = r'c:\Google Drive\Deltares\MWCI\01_data\water quality\Water Quality data_LLDA_v2_1999_2016.xlsx'
    
    datFile = pd.ExcelFile(excelFile)
    sheets = datFile.sheet_names
    dft = pd.DataFrame()
    
    
    # collect all locations
    for sheet in sheets:
        if 'Figures' not in sheet and 'PrimProd' not in sheet:
            dff = pd.read_excel(excelFile, sheetname = sheet)
            dff = dff[['Salinity (Chloride)','Date']]
            dff.rename(columns = {'Salinity (Chloride)' : 'Salinity ' + sheet}, inplace = True)
            dff.set_index('Date',inplace = True)
            dff.replace('-',np.nan, inplace = True)
            dff = dff[~dff.index.duplicated(keep='first')]
            dft = pd.concat([dff,dft], axis = 1, ignore_index = False)
    
    dff = dft.copy()
    meteo = r'c:\Google Drive\Deltares\MWCI\01_data\hydrology\Meteo and Streamflow Data\Luzon Meteo Data - PAGASA.xlsx'
    meteoFile = pd.ExcelFile(meteo)
    locs = ['NAIA (MIA)','AMBULONG']
    dfm  = pd.DataFrame()
    
    for ll in locs:
        metDat = pd.read_excel(meteo, sheetname = ll)
        metDat['Date'] = pd.Series()
        for ii,yy in enumerate(metDat['YEAR']):
            metDat['Date'] = pd.to_datetime(metDat[['YEAR','MONTH','DAY']])
    
        metDat.set_index('Date',inplace = True)
        metDat[metDat['RAINFALL'] == -2] = np.nan
        metDat[metDat['RAINFALL'] == 'T'] = np.nan
        dfm = pd.concat([dfm,metDat['RAINFALL']], axis = 1)
        dfm.rename(columns={'RAINFALL' : (('Rainfall %s [mm]') % ll)}, inplace = True)
    
    excelFile = r'c:\Google Drive\Deltares\MWCI\01_data\hydrology\Lake Level binary.csv'
    dfw = pd.read_csv(excelFile)
    
    csvfile = r'c:\Google Drive\Deltares\MWCI\01_data\waterlevel\manila_south_harbor_predic.csv'
    dfs = pd.read_csv(csvfile)
    
    dfw.replace('-',np.nan, inplace = True)
    dfw['water level (lake) [m]'] = dfw['Water Level'] - 10.47
    dfs['water level (ocean) [m]'] = dfs['water level [m]']
    
    df = pd.concat([dfs['time'].map(make_ts),dfs['water level (ocean) [m]'],dfw['water level (lake) [m]']], axis = 1)
    df.set_index('time',inplace = True)
    
    #df['gradient [m]'] = df['water level (lake) [m]'] - df['water level (ocean) [m]']
    df = pd.concat([df,pd.DataFrame(dff)], axis = 1, ignore_index = False)
    
    # df = pd.concat([df,pd.DataFrame(dff['Salinity (Chloride)'].dropna())], axis = 1, ignore_index = False)
    for ll in locs:
        df = pd.concat([df,dfm[(('Rainfall %s [mm]') % ll)]], axis = 1)
        
    # df.rename(columns={'Salinity (Chloride)' : 'Chloride Stn 5 (mg/l)'}, inplace = True)
    # df.to_csv(r'c:\Google Drive\Deltares\MWCI\01_data\DS\LagunaLake_allLoc.csv')
    # dfl = df[df['Salinity CentralBayStn4_1999-2016'].isnull() == False]
    
    ###########################################################################
    # create week delays
    df.rename(columns = {'water level (ocean) [m]' : 'ocean'}, inplace = True)
    df.rename(columns = {'water level (lake) [m]' : 'lake'}, inplace = True)
    df = df[df.index.isnull() == False]
    lake = {}
    backlog = 3
    interval = 7

    for ii in range(1,backlog + 1):
        lake['lake' + str(ii)] = []
    for ind,dw in enumerate(df['lake']):
        for ii in range(1,backlog + 1):
            if ind > (interval*ii)-2 and ind < len(df['lake'])- (interval*ii) - 1:
                lake['lake'+ str(ii)].append([df['lake'].iloc[ind-interval*ii],df['lake'].index[ind]])
            else:
                lake['lake'+ str(ii)].append([np.nan,df['lake'].index[ind]])
    for ii in range(1,backlog + 1):
        #lake['lake' + str(ii)] = np.array(lake['lake' + str(ii)])    
        tmp = pd.DataFrame(np.array(lake['lake' + str(ii)]))
        tmp.set_index(1,inplace = True)
        tmp.rename(columns = {0 : 'lake' + str(ii)}, inplace = True)
        df = pd.concat([df,tmp],  axis = 1)
        
    # interpolate salinity data
    # looped because interval between measurements is not constant
    # how many values to insert 
    div = 3
    ldat = df[df['Salinity WestBay_Stn1_1999-2016'].isnull() == False].index
    tind = [ind for ind,ll in enumerate(df.index) if ll in ldat]
    for col in df.columns:
        df[col].replace('*', np.nan,inplace = True)

    for ind,time in enumerate(ldat):
        if ind > 0:
            for col in df.columns:
                v2 = df[col].iloc[tind[ind]]
                v1 = df[col].iloc[tind[ind-1]]
                # choose the indicies that break the interval into the desired steps
                inter = int(np.floor((tind[ind] - tind[ind-1])/(div+1)))
                if inter != 0:
                    x = np.arange(tind[ind-1],tind[ind],inter)
                    arr = np.interp(x,[tind[ind-1],tind[ind]],[v1,v2])
                    df[col].iloc[x] = arr 
    for col in df.columns:
        if 'Central' in col or 'East' in col or 'South' in col or 'West' in col or 'lake' in col:
            pass
        else:
            df.drop(col, axis = 1, inplace = True)
    df.dropna(axis = 0, inplace = True)                           
    df.to_csv(r'c:\Google Drive\python\DDDSC\01_datasets\LagunaLake_clean.csv')

    ###########################################################################
if not process:
    file = r'c:\Google Drive\python\DDDSC\01_datasets\LagunaLake_clean.csv'
    df = pd.read_csv(file)
    df.set_index('Unnamed: 0',inplace = True)

    for col in df.columns:
        if 'Central' in col or 'East' in col or 'South' in col or 'West' in col or 'lake' in col:
            pass
        else:
            df.drop(col, axis = 1, inplace = True)

for col in df.columns:
    if '1999' in col and 'East' not in col:
        df.drop(col,inplace = True, axis = 1)
    else:
        # normalize
        df[col] = df[col] /df[col].sum()

model = keras.Sequential()
model.add(keras.layers.Dense(12, input_dim = 4, activation = 'tanh'))
#model_2.add(keras.layers.Dense(8, activation = 'tanh'))
