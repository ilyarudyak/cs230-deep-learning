import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

from course1_nn_dl.week2_log_regression.lr_utils import load_dataset


def get_processed_data():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # index = 10
    # plt.imshow(train_set_x_orig[index])
    # is_cat = train_set_y[0, index] == 1
    # title = 'this is cat' if is_cat else 'this is NOT cat'
    # plt.title(title)
    # plt.show()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    return train_set_x, train_set_y, test_set_x, test_set_y


def plot_sample_pictures():
    train_images, train_labels, _, _, _ = load_dataset()
    fig, axs = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(train_images[i])
        ax.legend(title=str(train_labels[0, i]), labels=['0'], fontsize='xx-small', loc='lower right')
        ax.set(xticks=[], yticks=[])
    plt.show()


def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w, b = np.zeros((dim, 1)), 0
    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    _, m = X.shape

    # forward prop
    A = sigmoid(w.T.dot(X) + b)
    A[A == 1.0] = .99999
    cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))

    # back prop
    dw = (1.0 / m) * X.dot((A - Y).T)
    db = (1.0 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {'dw': dw, 'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []

    for i in range(num_iterations):

        # propagate
        grads, cost = propagate(w, b, X, Y)
        dw, db = grads['dw'], grads['db']

        # update weights
        w -= learning_rate * dw
        b -= learning_rate * db

        # record the costs
        if i % 100 == 0:
            costs.append(cost)

        # print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print(f'Cost after iteration {i}: {float(cost):.2f}')

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    Y_prediction = w.T.dot(X) + b > 0
    return Y_prediction.astype(int)


def model(X_train, Y_train, X_test, Y_test,
          num_iterations=2000, learning_rate=0.5, print_cost=False):
    n, m = X_train.shape

    # initialize parameters
    w, b = initialize_with_zeros(n)

    # optimize model
    params, grads, costs = optimize(w, b, X_train, Y_train,
                                    num_iterations,
                                    learning_rate,
                                    print_cost=print_cost)
    w, b = params['w'], params['b']

    # predict
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    accuracy_train = accuracy(Y_prediction_train, Y_train)
    accuracy_test = accuracy(Y_prediction_test, Y_test)
    print(f'accuracy train:{accuracy_train:.6f}\naccuracy test:{accuracy_test:.6f}')

    d = {'num_iterations': num_iterations,
         'learning_rate': learning_rate,
         'w': w,
         'b': b,
         'Y_prediction_train': Y_prediction_train,
         'Y_prediction_test': Y_prediction_test,
         'costs': costs}

    return d


def accuracy(Y_pred, Y_true):
    return np.mean((Y_pred == Y_true).astype(float))


def plot_costs(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f'Learning rate = {learning_rate}')
    plt.show()


def get_wrong_images(n_images=10):
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    mistakes_mask = d['Y_prediction_test'] != Y_test
    mistakes_images = test_set_x_orig[mistakes_mask[0], :, :, :]
    return mistakes_images[0:n_images, :, :, :]


def plot_wrong_images(n_rows=2, n_cols=5, size=6):
    images = get_wrong_images()
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(size, size))
    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i], cmap='binary')
        axi.set(xticks=[], yticks=[])
    plt.show()


def get_prediction_on_image(image_filename, num_px=64):
    image = np.array(ndimage.imread(image_filename, flatten=False))
    image_reshaped = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
    y_pred = predict(d["w"], d["b"], image_reshaped)
    return y_pred


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_processed_data()
    d = model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.005,
              print_cost=True)

    # plot_costs(np.squeeze(d['costs']), d['learning_rate'])
    # plot_wrong_images()

    # fname = './my_cat_image.jpg'
    # image = np.array(ndimage.imread(fname, flatten=False))
    # plt.imshow(image)
    # plt.show()
    print(get_prediction_on_image('./my_cat_image.jpg'))

