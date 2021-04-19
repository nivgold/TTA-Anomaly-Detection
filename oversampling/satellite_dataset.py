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


class SatelliteDataset:
    def __init__(self, data_dir_path, from_disk=False, disk_path='/home/nivgold/pkls/oversampling_pkls/satellite_pkls'):
        self.disk_path = disk_path

        if from_disk:
            print("Loading pkls...")

            # loading from files
            self.train_features = pd.read_pickle(disk_path+"/train_features_satellite.pkl")
            self.train_labels = pd.read_pickle(disk_path+"/train_labels_satellite.pkl")

            self.test_features = pd.read_pickle(disk_path+"/test_features_satellite.pkl")
            self.test_labels = pd.read_pickle(disk_path+"/test_labels_satellite.pkl")

            self.features_full = pd.read_pickle(disk_path + "/features_full_satellite.pkl")
            self.labels_full = pd.read_pickle(disk_path + "/labels_full_satellite.pkl")

        else:
            file_name = 'satellite.csv'
            data_path = os.path.join(data_dir_path, file_name)
            print(f'Openning {file_name}')
            full_df = pd.read_csv(data_path)
            full_df = self.preprocessing(full_df)
            full_df = reduce_mem_usage(full_df)

            label_col = '36'

            self.labels_full = full_df[label_col]
            self.features_full = full_df.drop(columns=[label_col], axis=1)

            train_df, test_df = train_test_split(full_df)

            # TRAIN

            # filter out malicious aggregations (with label equal to 1)
            print(f'number of train samples: {len(train_df[train_df[label_col] == 0])}')
            train_df = train_df[train_df[label_col] == 0]
            self.train_labels = train_df[label_col]
            self.train_features = train_df.drop(columns=[label_col], axis=1)

            # TEST

            self.test_labels = test_df[label_col]
            self.test_features = test_df.drop(columns=[label_col], axis=1)

            # saving the attributes to disk
            self.save_attributes_to_disk()

    def preprocessing(self, data):

        def _normalize_columns(df):
            # min-max normalization
            scaler = MinMaxScaler()
            df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
            return df

        def _agg_df(df):
            target_col = '36'

            # df[target_col] = np.where(df[target_col] == 'normal', 0, 1)

            desired_cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']

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
        pd.to_pickle(self.train_features, self.disk_path+"/train_features_satellite.pkl")
        pd.to_pickle(self.train_labels, self.disk_path+"/train_labels_satellite.pkl")
        # save test
        pd.to_pickle(self.test_features, self.disk_path+"/test_features_satellite.pkl")
        pd.to_pickle(self.test_labels, self.disk_path+"/test_labels_satellite.pkl")
        # save train full
        pd.to_pickle(self.features_full, self.disk_path + "/features_full_satellite.pkl")
        pd.to_pickle(self.labels_full, self.disk_path + "/labels_full_satellite.pkl")


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
    satellite = SatelliteDataset(data_path, from_disk=from_disk)

    # train_ds = df_to_dataset(data=ids17.train_features, labels=ids17.train_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    train_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(satellite.train_features), satellite.train_labels))
                .cache()
                .batch(batch_size)
                .map(train_pack_features_vector))

    # test_ds = df_to_dataset(data=ids17.test_features, labels=ids17.test_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    test_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(satellite.test_features), satellite.test_labels))
               .cache()
               .batch(batch_size)
               .map(test_pack_features_vector))

    return train_ds, test_ds, satellite.features_full.values