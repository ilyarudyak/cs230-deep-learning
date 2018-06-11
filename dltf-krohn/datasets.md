- **MNIST database** (`LeNet`). Created by [Yann LeCun](http://yann.lecun.com/exdb/mnist/). 
The MNIST database contains 60,000 training images and 10,000 testing images.
ccFurthermore, the black and white images from NIST were normalized to fit 
into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale 
levels. Best models (CNN) have ~.20% error level. For comparison SVM gives .56%.

- **Oxford flowers** (`VGGNet`). Created in [Oxford](http://www.robots.ox.ac.uk/~vgg/data/flowers/).
We have created two flower datasets by gathering images from various websites, 
with some supplementary images from our own photographs. The first dataset is a 
smaller one consisting of 17 different flower categories, and the second dataset 
is much larger, consisting of 102 different categories of flowers common to the UK.

- **CIFAR-10/100**. Created by [Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton](https://www.cs.toronto.edu/~kriz/cifar.html).
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images. 

- **Imagenet** (`AlexNet`). Created by Fei Fei Li (?). Over 14 million URLs of images have been hand-annotated by ImageNet 
to indicate what objects are pictured; in at least one million of the images, bounding boxes are also provided.
ImageNet contains over 20 thousand categories; a typical category, such as "balloon" or "strawberry", 
contains several hundred images. 

A dramatic 2012 breakthrough in solving the ImageNet Challenge is widely considered to be the beginning of 
the deep learning revolution of the 2010s: "Suddenly people started to pay attention, not just within the 
AI community but across the technology industry as a whole."