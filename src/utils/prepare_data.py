import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
import json
names = ["class", "title", "content"]


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def convert_to_onehot(Y, n_class):
    return np.eye(n_class)[Y.reshape(-1)]


def load_data(file_name, sample_ratio=1, n_class=15, names=names, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["class"])
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y


def data_preprocessing(train, test, max_len):
    """transform to one-hot idx vector by VocabularyProcessor"""
    """VocabularyProcessor is deprecated, use v2 instead"""
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size


def data_preprocessing_v2(train, test, max_len, max_words=400000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2


def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        # words.add('PAD')
        words.add('<UNK>')
        # word_to_vec_map['PAD'] = np.zeros(100)
        word_to_vec_map['<UNK>'] = np.random.rand(100)
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in words:
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def load_data_simEval():
    train_X_list = []
    test_X_list = []
    train_Y_list = []
    test_Y_list = []
    with open('../data/train.json', 'r') as f_train:
        train = json.load(f_train)
        for idx, ele in enumerate(train):
            train_X_list.append(np.asarray(ele['sentence']))
            train_Y_list.append(ele['relation'])

    with open('../data/dev.json', 'r') as f_dev:
        test = json.load(f_dev)
        for idx, ele in enumerate(test):
            test_X_list.append(np.asarray(ele['sentence']))
            test_Y_list.append(ele['relation'])
    with open('../data/r2id.json', 'r') as f_r2id:
        r2id = json.load(f_r2id)
        id2r = {v: k for k, v in enumerate(r2id)}
    max_len = max(len(max(train_X_list, key=len)), len(max(test_X_list, key=len)))

    train_X = np.asarray(train_X_list)
    test_X = np.asarray(test_X_list)
    train_Y = np.array(train_Y_list, dtype=np.int32)
    test_Y = np.array(test_Y_list, dtype=np.int32)

    train_Y_oh = convert_to_onehot(train_Y, 10)
    test_Y_oh = convert_to_onehot(test_Y, 10)
    return train_X, test_X, train_Y, test_Y, r2id, id2r, max_len
# load_data_simEval()

