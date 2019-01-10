# Necessary librarys
import os # it's a operational system library, to set some informations
import random # random is to generate random values

import pandas as pd # to manipulate data frames 
import numpy as np # to work with matrix
from scipy.stats import kurtosis, skew # it's to explore some statistics of numerical values

import json # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file

columns = ['device', 'geoNetwork', 'totals', 'trafficSource'] # Columns that have json format

dir_path = "../" # you can change to your local 

# p is a fractional number to skiprows and read just a random sample of the our dataset. 
p = 0.07 # *** In this case we will use 50% of data set *** #


to_drop = ["socialEngagementType",'device.browserVersion', 'device.browserSize', 'device.flashVersion', 'device.language', 
                   'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel',
                              'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 
                                         'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude','geoNetwork.networkLocation', 
                                                    'trafficSource.adwordsClickInfo.criteriaParameters', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.campaign',
                                                               'trafficSource.adwordsClickInfo.page', 'trafficSource.referralPath', 'trafficSource.adwordsClickInfo.slot',
                                                                          'trafficSource.adContent', 'trafficSource.keyword']


#Code to transform the json format columns in table
def json_read(df):
    #joining the [ path + df received ]
    data_frame = dir_path + df

    #Importing the dataset
    df = pd.read_csv(data_frame, 
                    converters={column: json.loads for column in columns}, # loading the json columns properly
                    dtype={'fullVisitorId': 'str'}, # transforming this column to string
                    skiprows=lambda i: i>0 and random.random() > p)# Number of rows that will be imported randomly

    for column in columns:
        column_as_df = json_normalize(df[column]) 
        column_as_df.columns = ["{}.{}".format(column, subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        print(df.shape)

    return df # returning the df after importing and transforming


df_train = json_read("train_v2.csv")

df_train.drop(to_drop, axis=1, inplace=True)
df_train.head()
