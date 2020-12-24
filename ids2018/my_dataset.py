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


def train_test_split(full_df, test_ratio=0.3):
    test_last_idx = int(len(full_df) * test_ratio)
    train_df = full_df.iloc[test_last_idx:, :]
    test_df = full_df.iloc[:test_last_idx, :]

    return train_df, test_df


class IDS2018Dataset:
    def __init__(self, data_dir_path, from_disk=False):

        if from_disk:

            path = '/home/nivgold/TTA-Anomaly-Detection/ids2018/ids2018_out/ids2018_4tta_flow_id_rolling'
            # loading from files
            self.train_features = pd.read_pickle(path+"/train_features.pkl")
            self.train_labels = pd.read_pickle(path+"/train_labels.pkl")

            self.test_features = pd.read_pickle(path+"/test_features.pkl")
            self.test_labels = pd.read_pickle(path+"/test_labels.pkl")

            self.tta_features = list(np.load(path+"/tta_features.npy"))
            self.tta_labels = list(np.load(path+"/tta_labels.npy"))

            print(f'number of train samples: {len(self.train_labels)}')
            print(f'number of test samples: {len(self.test_labels)}')

        else:

            # trying opening only the 3.3G file
            print(f'Openning {data_dir_path}')
            full_df = pd.read_csv(data_dir_path, nrows=3000000)
            # full_df = full_df.dropna()
            full_df = self.preprocessing(full_df)
            full_df = reduce_mem_usage(full_df)

            train_df, test_df = train_test_split(full_df, 0.3)

            # TRAIN

            # taking every 5th sample
            train_df = train_df[5::5]

            # filter out malicious aggregations (with label equal to 1)
            print(f'number of train samples: {len(train_df[train_df.Label == 0])}')
            train_df = train_df[train_df['Label'] == 0]
            self.train_labels = train_df['Label']
            self.train_features = train_df.drop(columns=['Label'], axis=1)

            print(self.train_features)

            # TEST
            test_df = test_df.reset_index(drop=True)
            test_norm_df = test_df[5:-5:5]

            tta_labels = []
            tta_features = []

            # making the TTA data
            for index in test_norm_df.index:
                index = int(index)
                # take 4-before current index
                before_tta_samples = test_df.iloc[index - 4: index]
                # take 4-after current index
                after_tta_samples = test_df.iloc[index + 1: index + 5]
                # concat 4-before and 4-after df
                current_tta = pd.concat([before_tta_samples, after_tta_samples], axis=0)

                tta_labels.append(current_tta['Label'].values)
                tta_features.append(current_tta.drop(columns=['Label'], axis=1).values)

            # normal test data
            self.test_labels = test_norm_df['Label']
            self.test_features = test_norm_df.drop(columns=['Label'], axis=1)

            # tta data
            self.tta_labels = tta_labels
            self.tta_features = tta_features

            # saving the attributes to disk
            self.save_attributes_to_disk()



    def preprocessing(self, data):

        def _normalize_columns(df):
            df_labels = df['Label']
            df_features = df.drop('Label', axis=1)
            features = df_features.columns

            # min-max normalization
            scaler = MinMaxScaler()
            df_features_scaled = scaler.fit_transform(df_features.values)
            df_features_scaled = pd.DataFrame(df_features_scaled)
            df_features_scaled.columns = features
            return pd.concat([df_features_scaled, df_labels], axis=1)

        def _agg_df(df):
            # # converting the `Timestamp` column to be datetime dtype
            # try:
            #     df[' Timestamp'] = df[' Timestamp'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))
            # except Exception as e:
            #     df[' Timestamp'] = df[' Timestamp'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))

            target_col = 'Label'

            df[target_col] = np.where(df[target_col] == 'Benign', 0, 1)

            desired_cols = ['Flow Duration', 'Tot Fwd Pkts',
                               'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
                               'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
                               'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                               'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
                               'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
                               'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                               'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                               'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                               'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                               'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                               'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
                               'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
                               'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                               'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                               'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
                               'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
                               'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
                               'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
                               'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
                               'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']


            # df = df[desired_cols + [target_col]]
            df_features = df[['Flow ID'] + desired_cols]
            df_label = df[target_col]


            # # sliding window of 5 ticks
            # rolled_max = df_features.rolling(5).max()
            # rolled_max.columns = [f'MAX - {column}' for column in list(rolled_max.columns)]
            #
            # rolled_min = df_features.rolling(5).min()
            # rolled_min.columns = [f'MIN - {column}' for column in list(rolled_min.columns)]
            #
            # rolled_std = df_features.rolling(5).std()
            # rolled_std.columns = [f'STD - {column}' for column in list(rolled_std.columns)]
            #
            # rolled_labels = df_label.rolling(5).max()
            #
            # print('concatenating to create full rolled dataframe')
            # rolled = pd.concat([rolled_max, rolled_min, rolled_std, rolled_labels], axis=1)

            print("start group by flow id")
            # Trying to aggregate by Flow ID
            try:
                rolled_list = []
                grouped = df.groupby(by='Flow ID')
                for i, grouped_data in enumerate(grouped):
                    flow_id, flow_id_df = grouped_data
                    if i % 1000 == 0:
                        print(i)
                    # print(f'{flow_id}: len: {len(flow_id_df)}')
                    rolled_max = flow_id_df.rolling(5).max()
                    rolled_max.columns = [f'MAX - {column}' for column in list(rolled_max.columns)]

                    rolled_min = flow_id_df.rolling(5).min()
                    rolled_min.columns = [f'MIN - {column}' for column in list(rolled_min.columns)]

                    rolled_std = flow_id_df.rolling(5).std()
                    rolled_std.columns = [f'STD - {column}' for column in list(rolled_std.columns)]

                    rolled_labels = flow_id_df.rolling(5).max()

                    # print(rolled_max.shape, rolled_std.shape, rolled_min.shape, rolled_labels.shape)
                    current_rolled = pd.concat([rolled_max, rolled_min, rolled_std, rolled_labels], axis=1)
                    rolled_list.append(current_rolled)
            except Exception as e:
                print(e)

            # concatenate all the rolls to a total rolled
            print('concatenating to create full rolled dataframe')
            rolled = pd.concat(rolled_list, axis=0)

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

        path = '/home/nivgold/TTA-Anomaly-Detection/ids2018/ids2018_out/ids2018_4tta_flow_id_rolling'

        # save train
        pd.to_pickle(self.train_features, path+"/train_features.pkl")
        pd.to_pickle(self.train_labels, path+"/train_labels.pkl")
        # save test
        pd.to_pickle(self.test_features, path+"/test_features.pkl")
        pd.to_pickle(self.test_labels, path+"/test_labels.pkl")
        # save tta
        np.save(path+"/tta_features.npy", self.tta_features)
        np.save(path+"/tta_labels.npy", self.tta_labels)

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
    ids18 = IDS2018Dataset(data_path, from_disk=from_disk)

    # train_ds = df_to_dataset(data=ids17.train_features, labels=ids17.train_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    train_ds = (tf.data.Dataset.from_tensor_slices((dict(ids18.train_features), ids18.train_labels))
                .cache()
                .batch(batch_size)
                .map(train_pack_features_vector))

    # test_ds = df_to_dataset(data=ids17.test_features, labels=ids17.test_labels, shuffle=shuffle, batch_size=batch_size).map(pack_features_vector)
    test_ds = (tf.data.Dataset.from_tensor_slices(
        (dict(ids18.test_features), ids18.test_labels, ids18.tta_features, ids18.tta_labels))
               .cache()
               .batch(batch_size)
               .map(test_pack_features_vector))

    return train_ds, test_ds




"""## If Necessary - Create Tensorflow Dataset"""

# features_dict_values = [tf.float64 for i in range(features.shape[1])]
# features_dict_keys = list(features.columns)

# features_dict = dict(zip(features_dict_keys, features_dict_values))
# features_dict["label"] = tfds.features.ClassLabel(num_classes=2)


# class SSPD1DATASET(tfds.core.GeneratorBasedBuilder):
#   """SSPD Flooding Attack against the DVR Server"""

#   VERSION = tfds.core.Version('0.1.0')

#   def _info(self):
#     return tfds.core.DatasetInfo(
#         builder=self,
#         # This is the description that will appear on the datasets page.
#         description=("SSPD Flooding Attack against the DVR Server"),
#         # tfds.features.FeatureConnectors
#         features=tfds.features.FeaturesDict(features_dict),
#         # Homepage of the dataset for documentation
#         homepage="https://drive.google.com/drive/u/1/folders/1o3J-0uJ-ZhR_ETVANGPmnOBwx4332vlw",
#     )

#   def _split_generators(self, dl_manager):
#     # Downloads the data and defines the splits
#     # dl_manager is a tfds.download.DownloadManager that can be used to
#     # download and extract URLs
#     pass  # TODO

#   def _generate_examples(self):
#     # Yields examples from the dataset
#     yield 'key', {}
