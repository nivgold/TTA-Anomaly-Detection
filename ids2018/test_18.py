import ids2018.my_dataset as md
import os
import pandas as pd
import numpy as np

dir_path = '/home/nivgold/IT/Processed Traffic Data for ML Algorithms'


def clean_df(dirty_df):
    headers_rows = [-1] + list(dirty_df[dirty_df['Flow Duration'] == 'Flow Duration'].index)
    df_list = []
    for i in range(len(headers_rows) - 1):
        start = headers_rows[i]
        end = headers_rows[i + 1]
        df_list.append(dirty_df.loc[start + 1:end - 1])
    df_list.append(dirty_df.loc[headers_rows[-1] + 1:])
    final_df = pd.concat(df_list, axis=0)
    return final_df

file_path = dir_path + "/" + "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
# file_path = dir_path + "/" + "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
# df = pd.read_csv(file_path)
# print(df)
# print()
# print(df.columns)

ids18 = md.IDS2018Dataset(file_path, from_disk=False)