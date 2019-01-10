import pandas as pd
import numpy as np
import time

start = time.time()
train_df = pd.read_csv('./train_preprocessed.csv',dtype={'fullVisitorId': 'str'},)
train_df["totals.transactionRevenue"].fillna(0, inplace=True)
end = time.time()
print('loading complete')
print(end-start)

#train_df = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

#for value in train_df["totals.transactionRevenue"]:
 #   if value > 0:
  #      print(value)

avg_revenue = np.mean(train_df["totals.transactionRevenue"])
print(avg_revenue)

log_revenue = np.log1p(avg_revenue)
print(log_revenue)


