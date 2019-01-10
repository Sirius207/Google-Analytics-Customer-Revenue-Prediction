import pandas as pd
import time

ans = pd.read_csv('./data/baseline_lgb_const.csv', dtype={'fullVisitorId': 'str'})
zero = pd.read_csv('./old_user_id.csv', dtype={'fullVisitorId': 'str'})

start = time.time()
print(ans.shape)
print(zero.shape)
zero = zero.set_index('fullVisitorId')
zero = zero.to_dict('index')
end = time.time()
print(end-start)

start = time.time()
num = 0
for index, id in enumerate(ans['fullVisitorId']):
    if id not in zero:
        ans.loc[index, 'PredictedLogRevenue'] = 0
        num+=1

print(num)
end = time.time()
print('finish')
print(end-start)
ans.to_csv('baseline_lgb_const_rm_new.csv', index=False)

