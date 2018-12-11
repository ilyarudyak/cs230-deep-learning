## questions
- what should be the size of validation set? for big datasets (for example 1M examples) 
we have to choose 1% as a validation set (so 10K examples); this is very different from 
usual practice to use 20-30%; (Ng-strategy-1-06);
- how should we create validation set? using `train_test_split` from sklearn;
- what is the structure of `history`? `history.history.keys() ['acc', 'loss', 'val_acc', 'val_loss']`
     
## hypeparams
(1) size of mini-batch

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


