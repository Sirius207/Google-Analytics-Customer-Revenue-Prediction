import pandas as pd 
from pandas.io.json import json_normalize
from ast import literal_eval

MAXROWS = 1e5 # per CSV

for file in ['../preprocessed.csv']:
    chunk = pd.read_csv(file, usecols=[6], skiprows=0)
    
    chunk.columns = ['hits']
    chunk['hits'][chunk['hits'] == "[]"] = "[{}]"
    chunk['hits'] = chunk['hits'].apply(literal_eval).str[0]
    chunk = json_normalize(chunk['hits'])

    # Extract the product and promo names from the complex nested structure into a simple flat list:
    if 'product' in chunk.columns:
    #print(chunk['product'][0])
        chunk['v2ProductName'] = chunk['product'].apply(lambda x: [p['v2ProductName'] for p in x] if type(x) == list else [])
        chunk['v2ProductCategory'] = chunk['product'].apply(lambda x: [p['v2ProductCategory'] for p in x] if type(x) == list else [])
        del chunk['product']
    if 'promotion' in chunk.columns:
        chunk['promoId']  = chunk['promotion'].apply(lambda x: [p['promoId'] for p in x] if type(x) == list else [])
        chunk['promoName']  = chunk['promotion'].apply(lambda x: [p['promoName'] for p in x] if type(x) == list else [])
        del chunk['promotion']
    chunk.to_csv("./hits_train.csv", index=False)
    rows += len(chunk.index)
    print(rows)
