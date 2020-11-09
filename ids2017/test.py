import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

desired_cols = [' Flow Duration',
                    ' Total Fwd Packets', ' Total Backward Packets',
                    'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
                    ' Fwd Packet Length Max', ' Fwd Packet Length Min',
                    ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                    'Bwd Packet Length Max', ' Bwd Packet Length Min',
                    ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
                    ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
                    ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                    ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
                    ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
                    ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
                    ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
                    ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
                    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
                    'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
                    ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
                    ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
                    ' Average Packet Size', ' Avg Fwd Segment Size',
                    ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
                    ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
                    ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                    ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
                    'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
                    ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
                    ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
                    ' Idle Max', ' Idle Min']

target_col = ' Label'


def label_func(label_vector):
    # print(label_vector.unique())
    label_unique = label_vector.unique()
    if len(label_unique) == 1 and (0 in label_unique):
        label_col = pd.Series(0, index=['Label'], dtype=np.int16)
    else:
        label_col = pd.Series(1, index=['Label'], dtype=np.int16)
    return label_col


agg_dict = dict(zip(desired_cols, [['max', 'min', 'std'] for i in range(len(desired_cols))]))
agg_dict[target_col] = label_func


def save_df(df, name_file):
    # saving the rolled df
    df.to_pickle(f'/home/nivgold/pkls/{name_file}.pkl')


def run(path):

    df = pd.read_csv(path, encoding='cp1252')

    df = df.dropna()

    df[' Label'] = np.where(df[' Label'] == 'BENIGN', 0, 1)

    # converting the `Timestamp` column to be datetime dtype
    try:
        df[' Timestamp'] = df[' Timestamp'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))
    except Exception as e:
        df[' Timestamp'] = df[' Timestamp'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))

    # df = df.set_index(' Timestamp')
    # df = df.sort_index()

    df = df[desired_cols + [target_col]]

    # sliding window of 5 ticks
    rolled = df.rolling(5).agg(agg_dict)

    # fixing columns names
    new_columns = list(rolled.columns)
    new_columns = [(i + ' - ' + j.upper()).strip() for i, j in new_columns]
    new_columns[-1] = 'Label'
    rolled.columns = new_columns

    rolled = rolled.applymap(lambda x: np.nan_to_num(x))
    rolled['Label'] = rolled['Label'].astype(np.int16)

    return rolled

def main():
    HOME_DIR_PATH = '/home/nivgold'
    DATA_DIR_NAME = 'TrafficLabelling'
    DATA_PATH = os.path.join(HOME_DIR_PATH, DATA_DIR_NAME)

    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".csv"):
            file_path = os.path.join(DATA_PATH, file_name)
            name = file_name[:-14]
            # check if already been done
            if name+".pkl" not in os.listdir('/home/nivgold/pkls'):
                print(f'Making file: {name}...')
                start = time.time()
                rolled_df = run(file_path)
                end = time.time()
                save_df(rolled_df, name)
                print(f'Finished file: {name}')
                get_execute_time(start, end)
            else:
                print(f'skipping {name}')


if __name__ == '__main__':
    main()