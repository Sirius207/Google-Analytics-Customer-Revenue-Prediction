
import numpy as np
import pandas as pd

train_df = pd.read_csv('../data/test_preprocessed.csv',dtype={'fullVisitorId': 'str'})
train_df["totals.transactionRevenue"].fillna(0, inplace=True)

#answer_df = pd.read_csv('../sample_submission_v2.csv',dtype={'fullVisitorId': 'str'})

revenue_dict = dict()

revenue_column = 'totals.transactionRevenue'
#revenue_column = 'PredictedLogRevenue'

for i, fullVisitorId in enumerate(train_df['fullVisitorId']):
    if fullVisitorId not in revenue_dict:
        revenue_dict[fullVisitorId] =  train_df[revenue_column][i]
    else: 
        revenue_dict[fullVisitorId] += train_df[revenue_column][i]

print(len(revenue_dict))


# generate Ground truth
#data = {Id: np.log(value + 1) for Id, value in revenue_dict.items()}
#ground_truth = pd.DataFrame.from_dict(data, orient='index', columns=['PredictedLogRevenue'])
#ground_truth['fullVisitorId'] = ground_truth.index
#ground_truth = ground_truth[['fullVisitorId', 'PredictedLogRevenue']]
#ground_truth.to_csv('log_answer.csv', index=False)


sum = 0
for Id in revenue_dict:
    predict = np.log(revenue_dict[Id] + 1)
    sum += (predict-0)**2

print((sum/296530)**0.5)
