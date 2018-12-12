## questions
- what should be the size of validation set? for big datasets (for example 1M examples) 
we have to choose 1% as a validation set (so 10K examples); this is very different from 
usual practice to use 20-30%; (Ng-strategy-1-06);
- how should we create validation set? using `train_test_split` from sklearn;
- what is the structure of `history`? `history.history.keys() ['acc', 'loss', 'val_acc', 'val_loss']`
     
## ideas
- what about learning rate???
- increase validation set up to 20%;
- use batch normalization;

## hypeparams
(1) size of mini-batch
(2) number of epochs

### size of mini-batch
- typical sizes: 64-128-256-512; use search to understand how they reduce cost function
 (practical-aspects-02-02);
- it seems, large size DECREASES ability to generalize of the model; 

> These methods operate in a small-batch regime wherein a fraction of the training data, 
usually 32--512 data points, is sampled to compute an approximation to the gradient. 
It has been observed in practice that when using a larger batch there is a significant 
degradation in the quality of the model, as measured by its ability to generalize.
(see [here](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network/236393#236393)
and [here](https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch)).

### number of epochs
#### keras callbacks
- `EarlyStopping(monitor='val_loss', patience=1)` - stop training when a monitored quantity 
has stopped improving.
- `ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False)` - save the model after every 
epoch (or the best only if `save_best_only=True`).

> The examples so far have adopted the strategy of training for enough epochs that you begin overfitting,
 using the first run to figure out the proper number of epochs to train for, and then finally launching 
 a new training run from scratch using this optimal number (7.2.1 Chollet).
 
> You can use the EarlyStopping callback to interrupt training once a target metric being monitored 
has stopped improving for a fixed number of epochs. For instance, this callback allows you to 
interrupt training as soon as you start overfitting, thus avoiding having to retrain your model 
for a smaller number of epochs. This callback is typically used in combination with ModelCheckpoint, 
which lets you continually save the model during training (and, optionally, save only the current 
best model so far: the version of the model that achieved the best performance at the end of an epoch).
(7.2.1 Chollet).



