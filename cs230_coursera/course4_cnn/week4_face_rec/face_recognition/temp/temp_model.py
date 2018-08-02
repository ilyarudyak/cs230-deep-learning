# example of training a final classification model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler


def get_train_data():
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    scalar = MinMaxScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    return X, y


def get_test_data():
    X_new, y_new = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
    scalar = MinMaxScaler()
    scalar.fit(X_new)
    X_new = scalar.transform(X_new)
    return X_new, y_new


def fit_model(X, y):
    # define and fit the final model
    model = Sequential()
    model.add(Dense(4, input_dim=2, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=10, verbose=0)
    return model


if __name__ == '__main__':
    X, y = get_train_data()
    X_new, y_new = get_test_data()
    model = fit_model(X, y)

    print(X_new.shape)

    y_pred = model.predict_classes(X_new)
    print(y_pred, y_new)
