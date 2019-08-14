# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:43:39 2018

@author: schueder
"""

# prepare dat for AI

import numpy as np
import pandas as pd
import seaborn as sns
import pylab
import keras
import matplotlib.pyplot as plt
import os
import inspect
import glob

sns.set()
def MakeTS(var):
    return pd.Timestamp(var) 

process = False
train = True
# number of retrospective periods
backlog = 0
# length of period (days)
interval = 7
method = 'RNN'

'''
observations
longer time period allows one to obtain the bias in the signal arising from periods of
low water level
larger backlog or interval reduces the number of valid points
'''

station = 'WestBay_Stn1_'

print(inspect.getfile(inspect.currentframe())) 
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

print('begin processing')

if process:   
    excelFile = 'data/LagunaLake/water_quality/Water_Quality_data_LLDA_v2_1999_2016.xlsx'
    
    datFile = pd.ExcelFile(excelFile)
    sheets = datFile.sheet_names
    dft = pd.DataFrame()
    
    # collect all locations
    for sheet in sheets:
        if 'Figures' not in sheet and 'PrimProd' not in sheet:
            dff = pd.read_excel(excelFile, sheetname = sheet)
            dff = dff[['Salinity (Chloride)','Date']]
            dff.rename(columns = {'Salinity (Chloride)' : 'Salinity ' + sheet}, inplace = True)
            dff.rename(columns = {'Date' : 'date'}, inplace = True)
            dff.set_index('date',inplace = True)
            dff.replace('-',np.nan, inplace = True)
            dff = dff[~dff.index.duplicated(keep='first')]
            dft = pd.concat([dff,dft], axis = 1, ignore_index = False)
    
    dff = dft.copy()
    ###########################################################################

    meteo = 'data/LagunaLake/hydrology/Meteo and Streamflow Data/Luzon Meteo Data - PAGASA.xlsx'
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
    for ll in locs:
        df = pd.concat([df,dfm[(('Rainfall %s [mm]') % ll)]], axis = 1)
###########################################################################
    
    excelFile = 'data/LagunaLake/hydrology/Lake Level binary.csv'
    dfw = pd.read_csv(excelFile)
    dfw['date'] = dfw['date'].apply(MakeTS)
    dfw.set_index('date', inplace = True)
    dfw['water level (lake) [m]'] = dfw['Water Level'] - 10.47
    dfw.replace('-',np.nan, inplace = True)

    csvfile = 'data/LagunaLake/waterlevel/manila_south_harbor_predic.csv'
    dfo = pd.read_csv(csvfile)  
    dfo['water level (ocean) [m]'] = dfo['water level [m]']
    dfo['time'] = dfo['time'].apply(MakeTS)
    dfo.rename(columns = {'time' : 'date'}, inplace = True)
    dfo.set_index('date', inplace = True)
   
    df = pd.concat([dfo,dfw], axis = 1)
    
    df = pd.concat([df,dff], axis = 1, ignore_index = False)

###########################################################################
    for file in glob.glob(r'data\LagunaLake\hydrology\Meteo and Streamflow Data\*.csv'):
        dff = pd.read_csv(file)
        dff['date'] = dff['date'].apply(MakeTS)
        dff.set_index('date', inplace = True)  
        df = pd.concat([df,dff], axis = 1, ignore_index = False)
    
    ###########################################################################
    # create week delays
    df.rename(columns = {'water level (ocean) [m]' : 'ocean'}, inplace = True)
    df.rename(columns = {'water level (lake) [m]' : 'lake'}, inplace = True)
    df = df[df.index.isnull() == False]
    lake = {}
    if method == 'ANN':
        for ii in range(1,backlog + 1):
            lake['lake' + str(ii)] = []
        for ind,dw in enumerate(df['lake']):
            for ii in range(1,backlog + 1):
                # create each staggered array if index is larger than the distance back you will search
                if ind > (interval*ii)-2 and ind < len(df['lake'])- (interval*ii) - 1:
                    lake['lake'+ str(ii)].append([df['lake'].iloc[ind-interval*ii],df['lake'].index[ind]])
                else:
                    lake['lake'+ str(ii)].append([np.nan,df['lake'].index[ind]])
    
        for ii in range(1,backlog + 1):
            tmp = pd.DataFrame(np.array(lake['lake' + str(ii)]))
            tmp.set_index(1,inplace = True)
            tmp.rename(columns = {0 : 'lake' + str(ii)}, inplace = True)
            df = pd.concat([df,tmp],  axis = 1)

    ###########################################################################        
    # interpolate salinity data
    # looped because interval between measurements is not constant
    # how many values to insert? 

    div = 3
    ldat = df[df['Salinity WestBay_Stn1_1999-2016'].isnull() == False].index
    # indicies of valid measurements
    tind = [ind for ind,ll in enumerate(df.index) if ll in ldat]
    for col in df.columns:
        df[col].replace('*', np.nan,inplace = True)

    for ind, time in enumerate(ldat):
        # for each safe location
        if ind > 0:
            # if not the first location
            for col in df.columns: # interpolate for all columns
                v2 = df[col].iloc[tind[ind]]
                v1 = df[col].iloc[tind[ind-1]]
                # choose the indicies that break the interval into the desired steps
                #inter = int(np.floor((tind[ind] - tind[ind-1])/(div+1)))
                inter = 1
                if inter != 0:
                    x = np.arange(tind[ind-1],tind[ind],inter)
                    arr = np.interp(x,[tind[ind-1],tind[ind]],[v1,v2])
                    df[col].iloc[x] = arr 
                                            
    df.to_csv('data/LagunaLake_clean.csv')

###########################################################################
# load and filter as needed

file = 'data/LagunaLake_clean.csv'
df = pd.read_csv(file)
df['date'] = df['date'].apply(MakeTS)
df.set_index('date',inplace = True)

for cind, col in enumerate(df.columns):
    if 'water' in col or station in col or 'lake' in col or 'ocean' in col or cind >=14:
        # normalize
        #df[col] = (df[col] - df[col].mean()) / df[col].std()
        df[col] = df[col] / df[col].max()

    else:
        df.drop(col, axis = 1, inplace = True)


df.dropna(axis = 0, inplace = True)

###########################################################################

if train:
    target = df.pop([ii for ii in df. columns if station in ii][0])

    print('begin training')
    if method == 'ANN':
        df = df[[ii for ii in df.columns if 'lake' in ii].append('ocean')]
        model = keras.Sequential()
        model.add(keras.layers.Dense(2 * len(df.columns), input_dim = len(df.columns), activation = 'tanh'))
        model.add(keras.layers.Dense(2 * len(df.columns), activation = 'relu'))
        model.add(keras.layers.Dense(1, activation = 'relu'))

        model.compile(loss='mean_squared_error', optimizer = 'SGD', metrics=['accuracy', 'loss'])

        history = model.fit(df, target,  epochs=100, batch_size=50)

    elif method == 'RNN':
        # X must be of dim (samples, time steps, features)
        # rainfall series is not long enough to include
        #df = df[['ocean', 'lake']]
        
        X = np.array(df)
        X = np.reshape(X, (1, X.shape[0], X.shape[1]))
        # memory size
        ts = 60
        X_l = np.zeros((X.shape[1] , ts, len(df.columns))) * np.nan
        Y_l = np.zeros((X.shape[1])) * np.nan
        # create timesteps array
        for time in range(0, X_l.shape[0]):
            print(time)
            if time < ts-1:
                pass
            else:

                X_l[time,:,:] = X[0, (time+1-ts):time+1, :]
                # time at 60 will be the 61st number, and will not line up. Thus we take the 60th number at index 59
                Y_l[time] = target[time]
        # clip the skipped part
        X_l = X_l[ts-1:,:,:]
        Y_l = Y_l[ts-1:]
        pred_time = df.index[ts-1:]
      
        model = keras.Sequential()
        model.add(keras.layers.LSTM(100, input_shape = (X_l.shape[1], X_l.shape[2])))
        #model.add(keras.layers.LSTM(32))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1))

        model.compile(loss='mean_squared_error', optimizer = 'adam', metrics=['accuracy'])
        history = model.fit(X_l, Y_l ,  epochs=80)        
    
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'])

    #plt.figure(2)
    #plt.plot(df)

    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])               
    ax1.plot(target, 'ro', label = 'observed')
    ax1.plot(pred_time, model.predict(X_l), 'b-', label = 'neural net')
    ax1.legend()

    ax1.set_xlabel('time')
    ax1.set_ylabel('salinity [-]')
    plt.title(station)
    pylab.savefig(station + '.png')


