### caffe
- From U.C. Berkeley; written in C++; python interface;
- Model zoo (AlexNet, VGG, GoogLeNet, ResNet, plus others); 
- Readable source code;
- No need to write code!


- (+) Good for feed forward networks
- **(+) Good for fine tuning existing networks**
- **(+) Train models without writing any code!**
- (+) Python interface is pretty useful!
- (-) Need to write C++ / CUDA for new GPU layers
- **(-) Not good for recurrent networks**
- (-) Cumbersome for big networks (GoogLeNet, ResNet)

### torch
- From NYU + IDIAP;
- Written in C and Lua;
- Used a lot a Facebook, DeepMind;
- Model zoo (loadcaffe, GoogLeNet v1, GoogLeNet v3, ResNet);
- Readable source code;


(-) Lua
(-) Less plug-and-play than Caffe
(-)You usually write your own training code
**(+) Lots of modular pieces that are easy to combine**
(+) Easy to write your own layer types and **run on GPU**
(+) Most of the library code is in Lua, easy to read
**(+) Lots of pretrained models!**
**(-) Not great for RNNs**

### teano
- From Yoshua Bengio’s group at University of Montreal;
- Embracing computation graphs, symbolic computation (*predecessor* of tensorflow);
- High-level wrappers: Keras, Lasagne;


- (+) Python + numpy
- **(+) Computational graph is nice abstraction**
- **(+) RNNs fit nicely in computational graph**
- (-) Raw Theano is somewhat low-level
- **(+) High level wrappers (Keras, Lasagne) ease the pain**
- (-) Error messages can be unhelpful
- (-) Large models can have long compile times
- (-) Much “fatter” than Torch; more magic
- (-) Patchy support for pretrained models

### tensorflow
- From Google
- Very similar to Theano - all about computation graphs 
- Easy visualizations (TensorBoard)
- Multi-GPU and multi-node training

- (+) Python + numpy
- **(+) Computational graph abstraction, like Theano; great for RNNs**
- (+) Much faster compile times than Theano
- (+) Slightly more convenient than raw Theano?
- **(+) TensorBoard for visualization**
- **(+) Data AND model parallelism; best of all frameworks**
- (+/-) Distributed models, but not open-source yet
- (-) Slower than other frameworks right now
- (-) Much “fatter” than Torch; more magic
- **(-) Not many pretrained models**
 
 
 
 
 
 

 
 