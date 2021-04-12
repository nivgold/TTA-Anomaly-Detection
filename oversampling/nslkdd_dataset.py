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


class NSLKDDDataset:
    def __init__(self, data_dir_path, from_disk=False, disk_path='/home/nivgold/pkls/oversampling_pkls/nslkdd_pkls'):
        self.disk_path = disk_path

        if from_disk:
            print("Loading pkls...")

            # loading from files
            self.train_features = pd.read_pickle(disk_path+"/train_features_nslkdd.pkl")
            self.train_labels = pd.read_pickle(disk_path+"/train_labels_nslkdd.pkl")

            self.test_features = pd.read_pickle(disk_path+"/test_features_nslkdd.pkl")
            self.test_labels = pd.read_pickle(disk_path+"/test_labels_nslkdd.pkl")

            self.train_features_full = pd.read_pickle(disk_path+"/train_features_full_nslkdd.pkl")
            self.train_labels_full = pd.read_pickle(disk_path+"/train_labels_full_nslkdd.pkl")

        else:
            train_file_name = "KDDTrain+.txt"
            print(f'Openning {train_file_name}')
            data_path = os.path.join(data_dir_path, train_file_name)
            train_df = pd.read_csv(data_path, encoding='cp1252', header=None)
            train_df = train_df.dropna()
            train_df = self.preprocessing(train_df)
            train_df = reduce_mem_usage(train_df)

            # compute anomlay percentage
            label_col = 41
            labels_proportions = train_df[label_col].value_counts()
            self.anomaly_percentage = labels_proportions[1] / labels_proportions.sum()

            # TRAIN

            train_df_full = train_df.copy()
            self.train_labels_full = train_df_full[label_col]
            self.train_features_full = train_df_full.drop(columns=[label_col], axis=1)
            # filter out malicious aggregations (with label equal to 1)
            train_df = train_df[train_df[label_col] == 0]
            self.train_labels = train_df[label_col]
            self.train_features = train_df.drop(columns=[label_col], axis=1)

            # TEST

            test_file_name = "KDDTest+.txt"
            print(f'Openning {test_file_name}')
            data_path = os.path.join(data_dir_path, test_file_name)
            test_df = pd.read_csv(data_path, encoding='cp1252', header=None)
            test_df = test_df.dropna()
            test_df = self.preprocessing(test_df)
            test_df = reduce_mem_usage(test_df)

            self.test_labels = test_df[label_col]
            self.test_features = test_df.drop(columns=[label_col], axis=1)

            # saving the attributes to disk
            self.save_attributes_to_disk()

    # TODO: complete preprocessing for NLS-KDD Dataset
    def preprocessing(self, data):

        def _normalize_columns(df):
            # min-max normalization
            scaler = MinMaxScaler()
            df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
            return df

        def _agg_df(df):
            target_col = 41

            df[target_col] = np.where(df[target_col] == 'normal', 0, 1)

            desired_cols = list(set(np.arange(41)) - set([1, 2, 3]))

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
        pd.to_pickle(self.train_features, self.disk_path+"/train_features_nslkdd.pkl")
        pd.to_pickle(self.train_labels, self.disk_path+"/train_labels_nslkdd.pkl")
        # save test
        pd.to_pickle(self.test_features, self.disk_path+"/test_features_nslkdd.pkl")
        pd.to_pickle(self.test_labels, self.disk_path+"/test_labels_nslkdd.pkl")

        # save train full
        pd.to_pickle(self.train_features_full, self.disk_path+"/train_features_full_nslkdd.pkl")
        pd.to_pickle(self.train_labels_full, self.disk_path+"/train_labels_full_nslkdd.pkl")


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
    nsl_dataset = NSLKDDDataset(data_path, from_disk=from_disk)

    # train_ds = df_to_dataset(data=ids17.train_features, labels=ids17.train_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    train_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(nsl_dataset.train_features), nsl_dataset.train_labels))
                .cache()
                .batch(batch_size)
                .map(train_pack_features_vector))

    # test_ds = df_to_dataset(data=ids17.test_features, labels=ids17.test_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    test_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(nsl_dataset.test_features), nsl_dataset.test_labels))
               .cache()
               .batch(batch_size)
               .map(test_pack_features_vector))

    train_full_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(nsl_dataset.train_features_full), nsl_dataset.train_labels_full))
                .cache()
                .batch(batch_size)
                .map(train_pack_features_vector))

    return train_ds, test_ds, train_full_ds