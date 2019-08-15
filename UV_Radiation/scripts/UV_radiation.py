import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
import pylab
import pandas as pd
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import keras
import seaborn as sns
%matplotlib inline
sns.set()
# http://www.temis.nl/uvradiation/product/uvi-uvd.html
# https://www.epd.gov.hk/eia/register/report/eiareport/eia_1482008/EIA/html/Text/S6_WQ.htm

def make_ts(x):
    try: 
        return pd.Timestamp(x) 
    except:
        return np.nan
    
    
def check_na(var):
    if var == 'n/a':
        return np.nan
    else:
        return var
    
    
def split_time(var):
    var = str(var)
    year = var[0:4]
    month = var[4:6]
    day = var[6:8]
    hour = var[8:10]
    minute = var[10:12]
    try:
        return pd.Timestamp(year + '-' + month + '-' + day + ' ' + hour + ':'  + minute)
    except:
        return np.nan

    
def pdistf(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
plt.rc('font', family = 'serif', size = 8)

plt.close('all')

uv_index = 8.4

def index_to_intensity(uv_index):
    '''
    calculate the expected uv radiation based on the uv index
    this function takes the spectrum shown here:
    https://www.hko.gov.hk/wxinfo/uvindex/english/wisuvindex.htm
    and determines the scale necessary to apply to the spectrum
    to have the uv index match the fed index. This scale solution
    is then applied to the spectrum and summed to get the uv radiation
    a translation of index to uva is shown here
    https://www.hko.gov.hk/wxinfo/uvinfo/uvinfo_e.html
    
    '''
    # plausible distribution
    uv_wl = np.arange(290, 401)
    # units are mW / m2 nm
    intensity = np.array([[290, 0.001], [295, 1], [300, 10], [310, 200], [330, 800], [400, 820]])
    weight = np.array([[290, 1], [300, 1], [330, 0.001], [400, 0.0001]])

    uv_intensity = np.interp(uv_wl, intensity[:,0], np.log(intensity[:,1]))
    uv_weight    = np.interp(uv_wl, weight[:,0]   , np.log(weight[:,1]))

    uv_intensity = np.exp(uv_intensity)
    uv_weight    = np.exp(uv_weight)
    plt.plot(uv_wl, uv_intensity, label = 'irradiance')
    plt.plot(uv_wl, uv_weight, label = 'erythemal action spectrum')
    plt.yscale('log')
    rad = 0
    for ind, wl in enumerate(uv_wl):
        rad = rad + (uv_intensity[ind] * uv_weight[ind])

    index_0 = 0.04 * rad

    # if the spectra is always the same, then one can assume the only
    # difference between different UV_index values is the intensity
    # This can be pulled out of the integral and treated as a scaling factor 
    # on the integral of intensity * weight

    # uv_index    = 0.04  * np.sum(m1 * uv_intensity * uv_weight)
    # uv_index_0  = 0.04  * np.sum(     uv_intensity * uv_weight)
    # divide one equation by the other

    m = uv_index / index_0
    uv = m * uv_intensity

    return(0.001 * uv.sum())

''' 
a translation of index to uva is shown here
https://www.hko.gov.hk/wxinfo/uvinfo/uvinfo_e.html
for an index of 8.4, the UVA radiation was 49.8
the current translation is equal to test 
'''
test = index_to_intensity(uv_index)
plt.legend()
plt.xlabel('wavelength [nm]')
# 45.48
# this should be higher (~49), but there are of course numerous uncertainties

# radiation file
# this is direct solar radiation from KP station, which is close to HKO
solar_file = r'../data/KP_direct.csv'
solar_df = pd.read_csv(solar_file)
solar_df['time'] = solar_df['Date'].apply(make_ts)
solar_df.dropna(inplace = True, axis = 0)
solar_df['MJ/m2'] = solar_df['MJ/m2'].apply(lambda x: x.replace('*',''))
solar_df['MJ/m2'] = solar_df['MJ/m2'].apply(check_na)
solar_df['MJ/m2'] = solar_df['MJ/m2'].apply(lambda x: float(x))
# mega to kilo to joules per hour to per second to PAR
solar_df['rate'] = solar_df['MJ/m2'] * 1e6 * 0.45 /3600

# UV file
# not clear where this is taken
uv_file = r'../data/uv_15min/UV_df.csv'
uv_df = pd.read_csv(uv_file)
uv_df['time'] = uv_df['time'].apply(split_time)
uv_df.dropna(inplace = True, axis = 0)
uv_df['uv'] = uv_df['uv'].apply(lambda x: float(x))
#uv_df['uv'].iloc[[uv_df['uv'] > 90]] = np.nan
uv_df['uv'] = uv_df['uv'].apply(lambda x: np.nan if x > 90.0 else x)
uv_df.dropna(inplace = True, axis = 0)

# cloud file
cloud_file = r'../data/hko_cld.csv'
cloud_df = pd.read_csv(cloud_file)
cloud_df['time'] = cloud_df['time'].apply(make_ts)
cloud_df['hour'] = cloud_df['time'].apply(lambda x: x.hour)

cloud_df.dropna(inplace = True, axis = 0)
cloud_df['day'] =  cloud_df['time'].apply(lambda x: x.timetuple().tm_yday )
cloud_df['cloud'] = cloud_df['cloud'].apply(lambda x: float(x))
cloud_df.dropna(inplace = True, axis = 0)

plt.close('all')
# set indicies
# slow
try:
    solar_df.set_index('time', inplace = True)
    solar_df = solar_df[~solar_df.index.duplicated(keep='first')]
    cloud_df.set_index('time', inplace = True)
    uv_df.set_index('time', inplace = True)
except:
    pass

# concatenate
X = pd.concat([solar_df['rate'], cloud_df['cloud'], cloud_df['hour'], cloud_df['day'], uv_df['uv']], axis = 1, ignore_index = False)
# I have checked the concatenation validity on a random 2019 and a random 2012 datapoint. rate, cloud, and uv check out
# eliminate nighttime data
X['rate'] = X['rate'].apply(lambda x: np.nan if x == 0 else x)
X['uv'] = X['uv'].apply(lambda x: np.nan if x == 0 else x)

X.dropna(axis = 0, inplace = True)

Y = X[['uv','hour']].copy()
X_m = X[['day', 'cloud', 'rate', 'hour']].copy()
Y['uv_rate'] = Y['uv'].apply(index_to_intensity)
RadSurf = X_m['rate']