import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import datetime

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

def load_df(csv_path='./test_v2.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str', 'totals.hits': 'float', 'totals.pageviews': 'float', 'visitNumber': 'float', 'visitStartTime': 'float'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = ["{}.{}".format(column, subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Loaded {{ os.path.basename({}) }}. Shape: {{ {} }}".format(csv_path, df.shape))
    return df

def preprocess(train_df):
    """
    Drop const. columns
    """
    # train_df["totals.transactionRevenue"].fillna(0, inplace=True)
    to_drop = [col for col in train_df.columns if train_df[col].nunique() == 1]
    train_df.drop(to_drop, axis=1, inplace=True)
    #train_df['totals.transactionRevenue'].fillna(train_df['totals.transactionRevenue'].mean(), inplace=True)
    for col in train_df.select_dtypes(include=['float']):
        train_df[col].fillna(train_df[col].mean(), inplace=True)
    train_df.fillna(0, inplace=True)

    # train_df.drop(train_df[train_df["totals.transactionRevenue"]==0].index, inplace=True)

train_df = load_df('./train_v2.csv')
test_df = pd.read_csv('./data/test_preprocessed.csv')
preprocess(train_df)
# train_df["totals.transactionRevenue"].fillna(0, inplace=True)
train_y = train_df["totals.transactionRevenue"].values
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values


# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device.browser", 
        "device.deviceCategory", "device.operatingSystem", 
        "geoNetwork.city", "geoNetwork.continent", 
        "geoNetwork.country", "geoNetwork.metro",
        "geoNetwork.networkDomain", "geoNetwork.region", 
        "geoNetwork.subContinent", "trafficSource.adContent", 
        "trafficSource.adwordsClickInfo.adNetworkType", 
        "trafficSource.adwordsClickInfo.gclId", 
        "trafficSource.adwordsClickInfo.page", 
        "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
        "trafficSource.keyword", "trafficSource.medium", 
        "trafficSource.referralPath", "trafficSource.source"]
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime"]    
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)

    # Split the train dataset into development and valid based on time 
dev_df = train_df[train_df['date']<=20170531]
val_df = train_df[train_df['date']>20170531]
dev_y = np.log1p(dev_df["totals.transactionRevenue"].astype(float).values)
val_y = np.log1p(val_df["totals.transactionRevenue"].astype(float).values)

dev_X = dev_df[cat_cols + num_cols] 
val_X = val_df[cat_cols + num_cols] 
test_X = test_df[cat_cols + num_cols]

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
            "objective" : "regression",
            "metric" : "rmse", 
            "num_leaves" : 30,
            "min_child_samples" : 100,
            "learning_rate" : 0.1,
            "bagging_fraction" : 0.7,
            "feature_fraction" : 0.5,
            "bagging_freq" : 5,
            "bagging_seed" : 2018,
            "verbosity" : -1

            }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)


from sklearn import metrics
pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].astype(float).values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("fill_mean.csv", index=False)


