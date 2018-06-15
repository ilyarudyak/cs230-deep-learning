# sizes - raw

Number of training examples: m_train = 209
Number of testing examples: m_test = 50
Height/Width of each image: num_px = 64
Each image is of size: (64, 64, 3)
train_set_x shape: (209, 64, 64, 3)
train_set_y shape: (1, 209)
test_set_x shape: (50, 64, 64, 3)
test_set_y shape: (1, 50)

# sizes - processed

train_set_x_flatten shape: (12288, 209) - so it's (nx, m)
train_set_y shape: (1, 209) - so it's (1, m), we're using row vectors
test_set_x_flatten shape: (12288, 50)
test_set_y shape: (1, 50)