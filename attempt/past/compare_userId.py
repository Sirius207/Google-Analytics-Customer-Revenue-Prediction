import pandas as pd
import time

start = time.time()
train_df = pd.read_csv('../data/train_preprocessed.csv',dtype={'fullVisitorId': 'str'})
test = pd.read_csv('./data/log_answer.csv',dtype={'fullVisitorId': 'str'})

print(train_df.shape)
train_df = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
print(train_df.shape)


test = test.set_index('fullVisitorId')
test = test.to_dict('index')
end = time.time()
print(end-start)


start=time.time()
num = 0
for id in train_df['fullVisitorId']:
    if id in test:
        num+=1

print(num)
end = time.time()
print(end-start)
