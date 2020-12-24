import my_dataset as md
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
# df = pd.read_csv(file_path, nrows=3000000)
# print(df.describe())
# print(df.isnull().sum().sum())
# df = df.dropna()
#
# print(df.describe())
# print(df.isnull().sum().sum())
# df = df.dropna()
# #
# print(df.iloc[:900000]['Label'].value_counts())
# #
# target_col = 'Label'
# #
# df[target_col] = np.where(df[target_col] == 'Benign', 0, 1)
#
# desired_cols = ['Flow Duration', 'Tot Fwd Pkts',
#                                'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
#                                'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
#                                'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
#                                'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
#                                'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
#                                'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
#                                'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
#                                'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
#                                'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
#                                'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
#                                'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
#                                'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
#                                'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
#                                'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
#                                'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
#                                'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
#                                'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
#                                'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
#                                'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
#                                'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
#
#
# df_features = df[desired_cols]
# df_labels = df[target_col]
#
# rolled_std = df_features.rolling(5).std()
# rolled_std.columns = [f'STD - {column}' for column in list(rolled_std.columns)]
# rolled_min = df_features.rolling(5).min()
# rolled_min.columns = [f'MIN - {column}' for column in list(rolled_min.columns)]
# rolled_max = df_features.rolling(5).max()
# rolled_max.columns = [f'MAX - {column}' for column in list(rolled_max.columns)]
#
# rolled_label = df_labels.rolling(5).max()
#
# rolled = pd.concat([rolled_std, rolled_max, rolled_min, rolled_label], axis=1)
#
# del rolled_min
# del rolled_std
# del rolled_max
# del rolled_label
#
# # TODO: maybe not the best idea to fill all of the nan with mean
# rolled = rolled.fillna(rolled.mean())
# rolled[target_col] = rolled[target_col].astype(np.int16)
#
# from sklearn.preprocessing import MinMaxScaler
#
# df_labels = rolled['Label']
# df_features = rolled.drop('Label', axis=1)
# features = df_features.columns
#
# scaler = MinMaxScaler()
# df_features_scaled = scaler.fit_transform(df_features.values)
# df_features_scaled = pd.DataFrame(df_features_scaled)
# df_features_scaled.columns = features
# df_scaled = pd.concat([df_features_scaled, df_labels], axis=1)
#
# del df_labels
# del df_features
# del df_features_scaled
#
# df_scaled = md.reduce_mem_usage(df_scaled)
#
# train_df, test_df = md.train_test_split(df_scaled, 0.3)
#
# train_df = train_df[5::5]
# train_df = train_df[train_df['Label'] == 0]
# print(train_df)
#
# train_labels = train_df['Label']
# train_features = train_df.drop(columns=['Label'], axis=1)
#
# print(train_features)
# print(f'train_features nans: {train_features.isnull().sum().sum()}')
# print(train_labels)




ids18 = md.IDS2018Dataset(file_path, from_disk=False)
print(ids18.train_features)
print(ids18.train_features.isnull().sum().sum())
print(ids18.test_features)