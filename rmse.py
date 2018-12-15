
import pandas as pd
import numpy as np

def main(submission):
    ground_truth = pd.read_csv('./data/log_answer.csv')
    test_answer = pd.read_csv(submission)
    test_answer["PredictedLogRevenue"].fillna(0, inplace=True)
    
    print(ground_truth.shape)
    print(test_answer.shape)

    ground_truth = ground_truth.set_index('fullVisitorId')
    ground_truth.columns = ['TruthLogRevenue']
    test_answer = test_answer.set_index('fullVisitorId')


    ground_truth = ground_truth.join(test_answer)
    print(ground_truth.head())
    print(((ground_truth.TruthLogRevenue - ground_truth.PredictedLogRevenue) ** 2).mean() ** .5)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit',
                default='sample_submission_v2.csv',
                help='input your answerdata file name')

    args = parser.parse_args()
    main(args.submit)
