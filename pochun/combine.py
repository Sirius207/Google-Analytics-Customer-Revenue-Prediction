import pandas as pd


binary = pd.read_csv('./binary_r.csv', dtype={'fullVisitorId': 'str'})
adder = pd.read_csv('./lgb_r.csv', dtype={'fullVisitorId': 'str'})


adder = adder.set_index('fullVisitorId')
adder = adder.to_dict('index')

for index, id in enumerate(binary['fullVisitorId']):
    if binary.loc[index, 'PredictedLogRevenue'] > 0:
        print(id)
        if id in adder:
            binary.loc[index, 'PredictedLogRevenue'] = adder[id]['PredictedLogRevenue']

binary.to_csv('mix_binary_const.csv', index=False)


