from models import *
from keras.callbacks import History


class HyperParamSearcher:
    def __init__(self, models):
        self.models = models

    def batch_size_search(self, filename):
        val_accs = {}
        for model in self.models:
            hist: History = model.train_model_aug()
            val_acc = hist.history['val_acc']
            val_accs[model.batch_size] = val_acc

        df = pd.DataFrame(val_accs)
        df.to_csv(filename, index=False)


if __name__ == '__main__':
    batch_sizes = [64, 128, 256, 512]
    models = [MnistModel(name='krohn', epochs=50, batch_size=bs) for bs in batch_sizes]

    hps = HyperParamSearcher(models)
    hps.batch_size_search('hp/hp_batch_size.csv')
