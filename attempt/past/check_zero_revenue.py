import pandas as pd
import time

start = time.time()
train_df = pd.read_csv('../data/train_preprocessed.csv',dtype={'fullVisitorId': 'str'},)
train_df["totals.transactionRevenue"].fillna(0, inplace=True)
end = time.time()
print('loading complete')
print(end-start)

#zero_list = list()

train_df = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
print(train_df.shape)

#for id in train_df['totals.transactionRevenue']:
 #   zero_list.append(id)



#for index, revenue in enumerate(train_df['totals.transactionRevenue']):
 #   if revenue == 0:
  #      zero_list.append(train_df['fullVisitorId'][index])


#zero_df = pd.DataFrame(zero_list, columns=["fullVisitorId"])
train_df.to_csv("old_user_id.csv", index=False, columns=['fullVisitorId'])
