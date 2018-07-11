from data_manager import CifarDataManager
from layers_raw import *


class ConvNetCIFAR10:
    def __init__(self, image_shape=(32, 32, 3), n_classes=10):
        self.cifar10 = CifarDataManager()
        self.x = neural_net_image_input(image_shape)
        self.y = neural_net_label_input(n_classes)
        self.keep_prob = neural_net_keep_prob_input()

    def build_conv_net(self):
        pass

    def train_loop(self, steps=2500, minibatch_size=50):
        pass


if __name__ == '__main__':
    tf.reset_default_graph()
    cnn = ConvNetCIFAR10()
    cnn.train_loop(steps=500)
