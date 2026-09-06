
import numpy as np
from tcn1 import TCN
from keras import layers
import math
import pandas as pd
import keras as keras
from sklearn.preprocessing import MinMaxScaler

import os

bpath = os.getcwd()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'



class TCN_demo:
    def __init__(self,
                 window_size=10,
                 nb_filters=5,
                 kernel_size=5,
                 optimizer="Adam",
                 loss="mse",
                 epochs=100,
                 batch_size=5,
                 validation_split=0,
                 dilations=[int(math.pow(2, i + 1)) for i in range(4)]
                 ):
        self.scaler = None
        self.window_size = window_size
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dilations = dilations

    def load_data(self, path):
        df = pd.read_csv(path, header=0, index_col=[0])
        return df

    def maxmin_scaler(self, df):
        if self.scaler:
            df_s = self.scaler.transform(df.values)
        else:
            self.scaler = MinMaxScaler()
            df_s = self.scaler.fit_transform(df.values)
        return df_s

    def creat_feature(self, df_s, column_len):
        X = []
        label = []
        for i in range(len(df_s) - self.window_size):
            X.append(df_s[i:i + self.window_size, :].tolist())
            label.append(df_s[i + self.window_size, -7])

        X = np.array(X).reshape(-1, self.window_size * column_len, 1)
        label = np.array(label)

        return X, label

    def mape(self, pred, true):
        return np.mean(np.abs((true - pred) / true)) * 100

    def rmse(self, pred, true):
        return np.sqrt(np.mean((pred - true) ** 2))

    def do_train(self, path):
        df_train = self.load_data(path)
        df_s = self.maxmin_scaler(df_train)
        column_len = df_train.shape[1]
        x_train, y_train = self.creat_feature(df_s, column_len)

        inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2]), name='inputs')
        t = TCN(return_sequences=False, nb_filters=self.nb_filters, kernel_size=self.kernel_size,
                dilations=self.dilations)(inputs)
        x = layers.Dense(units=13, activation='sigmoid')(t)
        outputs = layers.Dense(units=1, activation='sigmoid')(x)

        tcn_model = keras.Model(inputs, outputs)
        tcn_model.compile(optimizer=self.optimizer, loss='mae', metrics=['mae'])

        tcn_model.fit(x_train, y_train, epochs=self.epochs, validation_split=self.validation_split,
                      batch_size=self.batch_size, verbose=0)
        tcn_model.summary()
        tcn_model.save(r'model_tcn\model1.h5')

    def do_predict(self, path):
        df_test = self.load_data(path)
        df_s = self.maxmin_scaler(df_test)
        column_len = df_test.shape[1]
        x_test, y_test = self.creat_feature(df_s, column_len)

        tcn_model = keras.models.load_model(r"model_tcn\model1.h5", custom_objects={'TCN': TCN})
        predict = tcn_model.predict(x_test)

        if predict.ndim == 1:
            predict = predict.reshape(-1, 1)

        pre_copies = np.repeat(predict, column_len, axis=-1)
        pred = self.scaler.inverse_transform(pre_copies)[:, -7]
        test_label = self.scaler.inverse_transform(np.repeat(y_test.reshape(-1, 1), column_len, axis=-1))[:, -7]

        mape_value = self.mape(pred, test_label)
        rmse_value = self.rmse(pred, test_label)

        return mape_value, rmse_value, pred, test_label


if __name__ == "__main__":
    demo = TCN_demo()
    my_path = ""
    demo.do_train(path=my_path)
    mape, rmse, pred, test_label = demo.do_predict(my_path)
    print(mape)
    print(rmse)