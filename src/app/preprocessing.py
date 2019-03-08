import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}
        self.splits = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(
            root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filename, filetype='csv', *, name, **kwargs):
        filepath = f'{self.directory}/{filename}'
        df = getattr(pd, f'read_{filetype}')(filepath, **kwargs)
        self.data[name] = df
        return df

    def save(self, name, filetype='csv', **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, **kwargs)

    def cleanup(self, name, *, drop=False, drop_duplicates=False):
        data = self.data[name]

        if drop is not False:
            data = data.drop(columns=drop)

        if drop_duplicates is True:
            data = data.drop_duplicates()

        self.data['clean'] = data

    def clean_dataset(self, name):
        data = self.data[name]
        data.dropna(inplace=True)
        indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
        self.data['clean'] = data[indices_to_keep].astype(np.float64)

    def one_hot_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data')
            return

        data = self.data['clean']
        categorical = pd.get_dummies(data[columns], dtype='int')
        data = data.drop(data[columns], axis=1)
        data = pd.concat([data, categorical], axis=1, sort=False)
        self.data['clean'] = data

        return data

    def label_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data')
            return
        data = self.data['clean']
        le = preprocessing.LabelEncoder()
        label = pd.DataFrame()
        stuff_to_label = columns
        ix = 0
        for i in stuff_to_label:
            le.fit(data[i])
            lab = le.transform(data[i])
            label.insert(ix, column=i, value=lab)
            ix += 1
        data = data.drop(stuff_to_label, axis=1)
        data = pd.concat([data, label], axis=1, sort=False)
        self.data['clean'] = data

        return data

    def split_data(self, *, target, size=0.2, state=42):
        if 'clean' not in self.data:
            print('Can not find clean data')
            return

        data = self.data['clean']
        y = np.array(data[target])
        X = np.array(data.drop(target, axis=1))
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=size, random_state=state)

        self.splits = {'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
                       'y_train': y_train, 'y_test': y_test, 'y_val': y_val}

        return X_train, X_test, X_val, y_train, y_test, y_val

    def scaler(self, *, scale_y=False):
        if 'X_train' not in self.splits:
            print('Splited data not available')
            return
        scaler = preprocessing.MinMaxScaler()
        self.splits['X_train'] = scaler.fit_transform(self.splits['X_train'])
        self.splits['X_test'] = scaler.transform(self.splits['X_test'])
        self.splits['X_val'] = scaler.transform(self.splits['X_val'])

        if scale_y:
            self.splits['y_train'] = scaler.fit_transform(
                self.splits['y_train'])
            self.splits['y_test'] = scaler.transform(self.splits['y_test'])
            self.splits['y_val'] = scaler.transform(self.splits['y_val'])

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value
