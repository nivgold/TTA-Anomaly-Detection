import pandas as pd
import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


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


class SatelliteDataset:
    def __init__(self, data_dir_path, from_disk=False, disk_path='/home/nivgold/pkls/oversampling_pkls/satellite_pkls'):
        self.disk_path = disk_path

        if from_disk:
            print("Loading pkls...")

            # loading from files
            self.features_full = pd.read_pickle(disk_path + "/features_full_satellite.pkl")
            self.labels_full = pd.read_pickle(disk_path + "/labels_full_satellite.pkl")

            self.X_pairs = np.load(self.disk_path + "/X_pairs_satellite.npy")
            self.y_pairs = np.load(self.disk_path + "/y_pairs_satellite.npy")

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

            # generating siamese pairs
            self.X_pairs, self.y_pairs = generate_pairs(self.features_full, self.labels_full)

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

        # save train full
        pd.to_pickle(self.features_full, self.disk_path + "/features_full_satellite.pkl")
        pd.to_pickle(self.labels_full, self.disk_path + "/labels_full_satellite.pkl")
        # save siamese pairs
        np.save(self.disk_path + "/X_pairs_satellite.npy", self.X_pairs)
        np.save(self.disk_path + "/y_pairs_satellite.npy", self.y_pairs)


def get_dataset(data_path, batch_size, from_disk=True):
    satellite = SatelliteDataset(data_path, from_disk=from_disk)

    return satellite.features_full.values, satellite.labels_full.values, (satellite.X_pairs, satellite.y_pairs)

def generate_pairs(features_full, labels_full):
    print(f"features full shape: {features_full.shape}, labels_full shape: {labels_full.shape}")
    data = pd.concat([features_full, labels_full], axis=1)
    label_col = '36'

    normal_data = data[data[label_col] == 0].drop(columns=[label_col], axis=1)
    anomaly_data = data[data[label_col] == 1].drop(columns=[label_col], axis=1)

    left_list = []
    right_list = []
    label_list = []

    print("pairing from normal data")
    for idx, row in tqdm(normal_data.iterrows(), total=len(normal_data)):
        same_sample = normal_data.sample(n=1).iloc[0]
        left_list.append(row)
        right_list.append(same_sample)
        label_list.append(1)
        different_sample = anomaly_data.sample(n=1).iloc[0]
        left_list.append(row)
        right_list.append(different_sample)
        label_list.append(0)
    
    print("pairing from anomaly data")
    for idx, row in tqdm(anomaly_data.iterrows(), total=len(anomaly_data)):
        same_sample = anomaly_data.sample(n=1).iloc[0]
        left_list.append(row)
        right_list.append(same_sample)
        label_list.append(1)
        different_sample = normal_data.sample(n=1).iloc[0]
        left_list.append(row)
        right_list.append(different_sample)
        label_list.append(0)

    left_list_data = pd.DataFrame(left_list).reset_index(drop=True)
    left_list_data.columns = [f'left_{col}' for col in left_list_data.columns]

    right_list_data = pd.DataFrame(right_list).reset_index(drop=True)
    right_list_data.columns = [f'right_{col}' for col in right_list_data.columns]

    label_df = pd.DataFrame({'label': label_list})

    full_df = pd.concat([left_list_data, right_list_data, label_df], axis=1)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    left_pairs = full_df.loc[:, left_list_data.columns]
    right_pairs = full_df.loc[:, right_list_data.columns]

    X = np.array([left_pairs.values, right_pairs.values])
    y = full_df['label'].values

    return X, y