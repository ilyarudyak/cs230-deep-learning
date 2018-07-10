from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    DATA_DIR = '/Users/ilyarudyak/data/mnist'
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    print(type(mnist), dir(mnist))
