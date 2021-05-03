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
    def __init__(self, data_dir_path, from_disk=False, disk_path='/home/nivgold/pkls/sliding_window_pkls/nslkdd_pkls'):
        self.disk_path = disk_path

        if from_disk:

            # loading from files
            self.train_features = pd.read_pickle(self.disk_path+"/train_features_nslkdd.pkl")
            self.train_labels = pd.read_pickle(self.disk_path+"/train_labels_nslkdd.pkl")

            self.test_features = pd.read_pickle(self.disk_path+"/test_features_nslkdd.pkl")
            self.test_labels = pd.read_pickle(self.disk_path+"/test_labels_nslkdd.pkl")

            self.tta_features = list(np.load(self.disk_path+"/tta_features_nslkdd.npy"))
            self.tta_labels = list(np.load(self.disk_path+"/tta_labels_nslkdd.npy"))

        else:
            train_file_name = "KDDTrain+.txt"
            print(f"Openning {train_file_name}")
            data_path = os.path.join(data_dir_path, train_file_name)
            train_df = pd.read_csv(data_path, encoding='cp1252', header=None)
            train_df = train_df.dropna()
            train_df = self.preprocessing(train_df)
            train_df = reduce_mem_usage(train_df)

            # TRAIN
            label_col = 41

            # taking every 5th sample
            train_df = train_df[5::5]

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

            # taking every 5th sample
            test_norm_df = test_df[5:-5:5]

            tta_labels = []
            tta_features = []

            print("Start making TTA samples...")
            # making the TTA data
            for index in test_norm_df.index:
                index = int(index)
                # take 4-before current index
                before_tta_samples = test_df.iloc[index - 4: index]
                # take 4-after current index
                after_tta_samples = test_df.iloc[index + 1: index + 5]
                # concat 4-before and 4-after df
                current_tta = pd.concat([before_tta_samples, after_tta_samples], axis=0)

                tta_labels.append(current_tta[label_col].values)
                tta_features.append(current_tta.drop(columns=[label_col], axis=1).values)

            # normal test data
            self.test_labels = test_norm_df[label_col]
            self.test_features = test_norm_df.drop(columns=[label_col], axis=1)

            # tta data
            self.tta_labels = tta_labels
            self.tta_features = tta_features

            # saving the attributes to disk
            self.save_attributes_to_disk()


    def preprocessing(self, data):

        def _normalize_columns(df):
            df_labels = df[41]
            df_features = df.drop(41, axis=1)
            features = df_features.columns

            # min-max normalization
            scaler = MinMaxScaler()
            df_features_scaled = scaler.fit_transform(df_features.values)
            df_features_scaled = pd.DataFrame(df_features_scaled)
            df_features_scaled.columns = features
            return pd.concat([df_features_scaled, df_labels], axis=1)

        def _agg_df(df):
            target_col = 41

            df[target_col] = np.where(df[target_col] == 'normal', 0, 1)

            desired_cols = list(set(np.arange(41)) - set([1, 2, 3]))

            # df = df[desired_cols + [target_col]]
            df_features = df[desired_cols]
            df_label = df[target_col]


            # sliding window of 5 ticks
            rolled_max = df_features.rolling(5).max()
            rolled_max.columns = [f'MAX - {column}' for column in list(rolled_max.columns)]

            rolled_min = df_features.rolling(5).min()
            rolled_min.columns = [f'MIN - {column}' for column in list(rolled_min.columns)]

            rolled_std = df_features.rolling(5).std()
            rolled_std.columns = [f'STD - {column}' for column in list(rolled_std.columns)]

            rolled_labels = df_label.rolling(5).max()

            print('concatenating to create full rolled dataframe')
            rolled = pd.concat([rolled_max, rolled_min, rolled_std, rolled_labels], axis=1)

            # TODO: maybe not the best idea to fill all of the nan with mean
            rolled = rolled.fillna(rolled.mean())
            rolled[target_col] = rolled[target_col].astype(np.int16)

            return rolled

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
        # save tta
        np.save(self.disk_path+"/tta_features_nslkdd.npy", self.tta_features)
        np.save(self.disk_path+"/tta_labels_nslkdd.npy", self.tta_labels)

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


def test_pack_features_vector(features, labels, tta_features, tta_labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels, tta_features, tta_labels


def get_dataset(data_path, batch_size, from_disk=True):
    # load the features and the labels and convert them to tf.Dataset (train and test)
    nslkdd = NSLKDDDataset(data_path, from_disk=from_disk)

    # train_ds = df_to_dataset(data=ids17.train_features, labels=ids17.train_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    train_ds = (tf.data.Dataset.from_tensor_slices((dict(nslkdd.train_features), nslkdd.train_labels))
                .cache()
                .batch(batch_size)
                .map(train_pack_features_vector))

    # test_ds = df_to_dataset(data=ids17.test_features, labels=ids17.test_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    test_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(nslkdd.test_features), nslkdd.test_labels, nslkdd.tta_features, nslkdd.tta_labels))
               .cache()
               .batch(batch_size)
               .map(test_pack_features_vector))

    return train_ds, test_ds