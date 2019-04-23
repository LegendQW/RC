import numpy as np
from keras.layers.embeddings import Embedding
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step,
        'pred_y': model.prediction,
        'embmatrix': model.batch_embed_show,
        'batch_idx': model.batch_idx,
        'mask_matrix': model.mask,
        'len': model.len_scope

    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction, loss = sess.run([model.prediction, model.loss], feed_dict)
    acc = float(np.sum(np.equal(prediction, batch[1]))) / len(prediction)
    # print(prediction == batch[1])
    return acc, loss, prediction


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]

    X_indices = np.zeros([m, max_len])
    for i in range(m):
        sentence_words = X[i]
        j = 0
        for w in sentence_words:
            if word_to_index.has_key(w):
                X_indices[i, j] = word_to_index[w]
            else:
                X_indices[i, j] = word_to_index['<UNK>']
            j = j + 1
        # X_indices[i, j:] = word_to_index['PAD']
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_dim):
    vocab_len = len(word_to_index) + 1
    emb_matrix = np.zeros([vocab_len, emb_dim])

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    #
    # embedding_layer.build((None,))
    # embedding_layer.set_weights([emb_matrix])
    # return embedding_layer
    # emb_matrix = np.vstack([np.zeros(100), emb_matrix])
    return emb_matrix


def mask_for_softmax(val, mask):
    return -1e30 * (1 - tf.cast(mask, tf.float32)) + val


def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0], ), rownames=['Actual'], colnames=['Predicted'],
                               margins=True)

    df_conf_norm = df_confusion / df_confusion.sum(axis=1)

    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)