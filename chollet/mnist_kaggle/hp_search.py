from models import *
from keras.callbacks import History


class HyperParamSearcher:
    def __init__(self, models):
        self.models = models

    def batch_size_search(self):
        val_accs = {}
        val_accs_max = {}
        for model in self.models:
            hist: History = model.train_model_aug()
            val_acc = hist.history['val_acc']
            val_accs[model.batch_size] = val_acc
            val_accs_max[model.batch_size] = max(val_acc)
        return val_accs, val_accs_max
            

if __name__ == '__main__':
    batch_sizes = [64, 128, 256, 512]
    models = [MnistModel(name='krohn', epochs=1, batch_size=bs, verbose=0) for bs in batch_sizes]

    hps = HyperParamSearcher(models)
    val_accs, val_accs_max = hps.batch_size_search()
    print(val_accs, val_accs_max)
