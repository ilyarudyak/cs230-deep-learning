import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, SGDRegressor

N_ROWS = 10_000_000
DATA_PATH = '~/data/nyc_taxi/train.csv'
FILENAMES = ['train.npy', 'train_b.npy', 'train_labels.npy']


def get_processed_data():
    train_df = pd.read_csv(DATA_PATH, nrows=N_ROWS)
    add_travel_vector_features(train_df)
    train_df = train_df.dropna(how='any', axis='rows')
    train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
    X_train = get_input_matrix(train_df)
    X_train_b = get_input_matrix_bias(train_df)
    y_train = np.array(train_df['fare_amount'])
    return X_train, X_train_b, y_train


def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


def get_input_matrix_bias(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))


def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude))


def get_data_from_file():
    train_file, train_bias_file, label_file = FILENAMES
    return np.load(train_file), np.load(train_bias_file), np.load(label_file)


def normal_eq_numpy(X_train_b, y_train):
    W, _, _, _ = np.linalg.lstsq(X_train_b, y_train)
    return W


def normal_eq(X_train_b, y_train):
    W = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
    return W


def normal_eq_sklearn(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr.coef_, lr.intercept_


def sgd_sklearn(X_train, y_train):
    sgd = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgd.fit(X_train, y_train.ravel())
    return sgd.coef_, sgd.intercept_


if __name__ == '__main__':
    X_train, X_train_b, y_train = get_data_from_file()
    # W = normal_eq_numpy(X_train_b, y_train)
    # W = normal_eq(X_train_b, y_train)
    # W, b = normal_eq_sklearn(X_train, y_train)
    W, b = sgd_sklearn(X_train, y_train)
    print(W, b)



