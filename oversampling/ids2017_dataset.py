import pandas as pd
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

AUTOTUNE = tf.data.experimental.AUTOTUNE


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def train_test_split(full_df, train_ratio=0.7):
    train_last_idx = int(len(full_df) * train_ratio)
    train_df = full_df.iloc[:train_last_idx, :]
    test_df = full_df.iloc[train_last_idx:, :]

    return train_df, test_df


class IDS2017Dataset:
    def __init__(self, data_dir_path, from_disk=False, disk_path='/home/nivgold/pkls/oversampling_pkls/ids2017_pkls'):
        self.disk_path = disk_path

        if from_disk:
            print("Loading pkls...")

            # loading from files
            self.train_features = pd.read_pickle(disk_path+"/train_features_ids2017.pkl")
            self.train_labels = pd.read_pickle(disk_path+"/train_labels_ids2017.pkl")

            self.test_features = pd.read_pickle(disk_path+"/test_features_ids2017.pkl")
            self.test_labels = pd.read_pickle(disk_path+"/test_labels_ids2017.pkl")

        else:
            df_list = []
            for file_name in os.listdir(data_dir_path):
                if file_name.endswith('.csv'):
                    print(f'Openning {file_name}')
                    data_path = os.path.join(data_dir_path, file_name)
                    df = pd.read_csv(data_path, encoding='cp1252')
                    df = df.dropna()
                    df_list.append(self.preprocessing(df))

            # concat rows
            full_df = pd.concat(df_list, axis=0)
            full_df = reduce_mem_usage(full_df)

            self.full_df = full_df

            # compute anomlay percentage
            label_col = full_df[' Label']
            labels_proportions = label_col.value_counts()
            self.anomaly_percentage = labels_proportions[1] / labels_proportions.sum()

            train_df, test_df = train_test_split(full_df, 0.7)

            # TRAIN

            # filter out malicious aggregations (with label equal to 1)
            train_df = train_df[train_df[' Label'] == 0]
            self.train_labels = train_df[' Label']
            self.train_features = train_df.drop(columns=[' Label'], axis=1)

            # TEST

            self.test_labels = test_df[' Label']
            self.test_features = test_df.drop(columns=[' Label'], axis=1)

            # saving the attributes to disk
            self.save_attributes_to_disk()

    def preprocessing(self, data):

        def _normalize_columns(df):
            # min-max normalization
            scaler = MinMaxScaler()
            df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
            return df

        def _agg_df(df):
            target_col = ' Label'

            df[target_col] = np.where(df[target_col] == 'BENIGN', 0, 1)

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

            df = df[desired_cols + [target_col]]

            # replacing inf values with column max
            df = df.replace([np.inf], np.nan)
            df = df.fillna(df.max())

            return df


        # ------ start of the preprocessing func ------
        print('aggregating...')
        data_df = _agg_df(data)
        print('normalizing...')
        data_df = _normalize_columns(data_df)

        print('finished preprocessing')

        return data_df

    def save_attributes_to_disk(self):
        print('saving attributes...')

        # save train
        pd.to_pickle(self.train_features, self.disk_path+"/train_features_ids2017.pkl")
        pd.to_pickle(self.train_labels, self.disk_path+"/train_labels_ids2017.pkl")
        # save test
        pd.to_pickle(self.test_features, self.disk_path+"/test_features_ids2017.pkl")
        pd.to_pickle(self.test_labels, self.disk_path+"/test_labels_ids2017.pkl")


def df_to_dataset(data, labels, shuffle=False, batch_size=32):
    # df -> dataset -> cache -> shuffle -> batch -> prefetch
    data = data.copy()
    labels = labels.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(data), labels)).cache()
    if shuffle:
        DATASET_SIZE = tf.data.experimental.cardinality(ds).numpy()
        ds = ds.shuffle(buffer_size=DATASET_SIZE)
    ds = ds.batch(batch_size)
    return ds


def train_pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def test_pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def get_dataset(data_path, batch_size, from_disk=True):
    # load the features and the labels and convert them to tf.Dataset (train and test)
    ids17 = IDS2017Dataset(data_path, from_disk=from_disk)

    # train_ds = df_to_dataset(data=ids17.train_features, labels=ids17.train_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    train_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(ids17.train_features), ids17.train_labels))
                .cache()
                .batch(batch_size)
                .map(train_pack_features_vector))

    # test_ds = df_to_dataset(data=ids17.test_features, labels=ids17.test_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    test_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(ids17.test_features), ids17.test_labels))
               .cache()
               .batch(batch_size)
               .map(test_pack_features_vector))

    return train_ds, test_ds