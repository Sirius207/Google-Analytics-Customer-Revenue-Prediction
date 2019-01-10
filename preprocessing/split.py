import pandas as pd
from pandas.io.json import json_normalize
import json
import time

def load_df(csv_path='./test_v2.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = ["{}.{}".format(column, subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Loaded {{ os.path.basename({}) }}. Shape: {{ {} }}".format(csv_path, df.shape))
    return df

def preprocess(train_df):
    """
    Drop const. columns and zero revenue rows
    """
    to_drop = [col for col in train_df.columns if train_df[col].nunique() == 1]
    train_df.drop(to_drop, axis=1, inplace=True)

    train_df["totals.transactionRevenue"].fillna(0, inplace=True)
    train_df.drop(train_df[train_df["totals.transactionRevenue"]==0].index, inplace=True)

#print "Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns))

def main():
    # start = time.time()
    # train_df = load_df("./split/train_v2_split00.csv")
    # end = time.time()
    # print end-start
    in_dir = './split/'
    out_dir = './split/drop/'

    split_no = 16
    for x in xrange(16):
        no = "%02d" % x
        infile = "train_v2_split{}.csv".format(no)
        outfile = "train_v2_split{}_drop.csv".format(no)

        print "Loading: {}".format(in_dir + infile)
        train_df = load_df(in_dir + infile)
        preprocess(train_df)
        print "Output: {}".format(out_dir + outfile)
        train_df.to_csv(out_dir + outfile, encoding='utf-8')

if __name__ == "__main__":
    main()
