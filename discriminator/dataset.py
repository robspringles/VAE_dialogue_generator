import os
import ast
import torch
from torchtext import data, datasets
from torchtext.vocab import GloVe
from collections import defaultdict

START_WORD = '<s>'
END_WORD = '</s>'
PAD_WORD = '<blank>'

class CLS(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            subtrees: Whether to include sentiment-tagged subphrases
                in addition to complete examples. Default: False.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: False.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
	    return {'0': 'irrelative', '1': 'relative', None: None}[label]
        label_field.preprocessing = data.Pipeline(get_label_str)

	with open(os.path.expanduser(path)) as f:
	    examples = [data.Example.fromlist(line.strip().split("\t"), fields) for line in f]

        super(CLS, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, path='./data',
               train='train.disc', validation='dev.disc', test='test.disc',
               train_subtrees=False, **kwargs):
        """Create dataset objects for splits of the SST dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.txt'.
            train_subtrees: Whether to use all subtrees in the training set.
                Default: False.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """Creater iterator objects for splits of the SST dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)

class CLS_Dataset:

    def __init__(self, emb_dim=128, mbsize=128):
        self.text_field = data.Field(pad_token=PAD_WORD, lower=True)
        self.label_field = data.Field(sequential=False, unk_token=None)

	train, val, test = CLS.splits(self.text_field, self.label_field)

	vocab = dict(torch.load("../data/dialogue.vocab.pt", "text"))
	v = vocab['src']; v.stoi = defaultdict(lambda: 0, v.stoi)
	self.text_field.vocab = v
	self.label_field.build_vocab(test)

        self.n_vocab = len(self.text_field.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1,
            shuffle=True, repeat=True)
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)

    def get_vocab_vectors(self):
        return self.text_field.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.text_field.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.label_field.vocab.itos[idx]


if __name__ == '__main__':
    CLS_Dataset()
