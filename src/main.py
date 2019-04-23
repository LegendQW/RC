# coding=utf-8

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from utils.prepare_data import *
import time
from utils.model_helper import *
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import json


class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        # self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]
        self.l2_loss = config["l2_loss"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph")
        # Word embedding
        # embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                              trainable=True)
        self.x = tf.cast(self.x, tf.int32)
        # for row in self.x:
        #     for item in row:

        self.mask = (~tf.equal(self.x, 0))
        len = tf.reduce_sum(tf.cast(self.mask, tf.float32), axis=1)
        len = tf.cast(len, tf.int32)
        self.len_scope = len
        # self.mask = tf.cast(self.mask, tf.float32)
        embeddings_var = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_dim=self.embedding_size)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        batch_embedded = tf.cast(batch_embedded, tf.float32)
        self.batch_embed_show = batch_embedded
        self.batch_idx = self.x

        self.total_emb_matrix = embeddings_var
        batch_embedded_drop = tf.nn.dropout(batch_embedded, keep_prob=0.7)

        fwLSTM = BasicLSTMCell(self.hidden_size)
        bwLSTM = BasicLSTMCell(self.hidden_size)
        fwLSTM_drop = tf.contrib.rnn.DropoutWrapper(fwLSTM, output_keep_prob=0.7)
        bwLSTM_drop = tf.contrib.rnn.DropoutWrapper(bwLSTM, output_keep_prob=0.7)
        rnn_outputs, _ = bi_rnn(fwLSTM_drop,
                                bwLSTM_drop, sequence_length=len,
                                inputs=batch_embedded_drop, dtype=tf.float32)

        # rnn_outputs_drop = tf.nn.dropout(rnn_outputs, keep_prob=0.3)
        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        before_mask = tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                           tf.reshape(W, [-1, 1])),
                                 (-1, self.max_len))
        after_mask = mask_for_softmax(before_mask, mask=self.mask)
        self.alpha = tf.nn.softmax(after_mask)  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat,
                                                           labels=self.label)) + tf.contrib.layers.l2_regularizer(
            self.l2_loss)(FC_W)

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')

        print("graph built successfully!")


if __name__ == '__main__':
    # load data
    # x_train, y_train = load_data("../dbpedia_data/dbpedia_csv/train.csv", sample_ratio=1e-2, one_hot=False)
    # x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/test.csv", one_hot=False)

    total_train_loss = []
    total_eval_loss = []
    f1_train_list = []
    f1_eval_list = []
    pred_test_list = []
    attn_dic = {}
    attn_list = []
    test_sentence_list = []
    test_result_list = []
    test_true_list = []
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../glove/glove.6B.100d.txt')

    train_X, test_X, train_Y, test_Y, r2id, id2r, max_len = load_data_simEval()

    config = {
        "max_len": max_len,
        "hidden_size": 64,
        # "vocab_size": vocab_size,
        "l2_loss": 1e-5,
        "embedding_size": 100,
        "n_class": 10,
        "learning_rate": 1.0,
        "batch_size": 10,
        "train_epoch": 1
    }

    # data preprocessing
    # x_train, x_test, vocab_size = data_preprocessing_v2(x_train, x_test, max_len=32)
    # print("train size: ", len(x_train))
    # print("vocab size: ", vocab_size)

    # split dataset to test and dev

    train_X_indices = sentences_to_indices(train_X, word_to_index, max_len=max_len)
    test_X_indices = sentences_to_indices(test_X, word_to_index, max_len=max_len)
    # train_X_indices = tf.cast(train_X_indices, tf.int32)
    # test_X_indices = tf.cast(test_X_indices, tf.int32)

    # embeddings_var_spec = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_dim=config['embedding_size'])
    # batch_embedded_spec = tf.nn.embedding_lookup(embeddings_var_spec, train_X_indices_tf[0])
    # with tf.Session() as sess:
    #     temp = sess.run(batch_embedded_spec[0])
    # print(temp)
    # print (word_to_vec_map[index_to_word[train_X_indices[0][0]]])
    x_test, x_dev, y_test, y_dev, dev_size, test_size = split_dataset(test_X_indices, test_Y, 0.5)
    print("Validation Size: ", dev_size)

    classifier = ABLSTM(config)
    classifier.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dev_batch = (x_dev, y_dev)
    start = time.time()
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        train_loss_list = []
        f1_batch_train_y = []
        for x_batch, y_batch in fill_feed_dict(train_X_indices, train_Y, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            batch_emb = return_dict['embmatrix']
            # print(batch_emb[0][0])
            len_scope = return_dict['len']
            batch_idx = return_dict['batch_idx']
            mask_matrix = return_dict['mask_matrix']
            total_emb = classifier.total_emb_matrix
            f1_batch_train_y.append(metrics.f1_score(y_batch, return_dict['pred_y'], average='weighted'))
            train_loss_list.append(return_dict['loss'])
            attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
            # plot the attention weight
            # print(np.reshape(attn, (config["batch_size"], config["max_len"])))
        t1 = time.time()
        train_loss = float(np.sum(np.asarray(train_loss_list))) / len(train_loss_list)
        f1_train_list.append(float(np.sum(np.asarray(f1_batch_train_y))) / len(f1_batch_train_y))
        total_train_loss.append(train_loss)
        print ("train_loss:{}".format(train_loss))
        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc, eva_loss, pred_eval = run_eval_step(classifier, sess, dev_batch)
        f1_eval_list.append(metrics.f1_score(dev_batch[1], pred_eval, average='weighted'))
        total_eval_loss.append(eva_loss)
        print ("eval_loss:{}".format(eva_loss))
        print("validation accuracy: %.3f " % dev_acc)

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc, loss, pred_test = run_eval_step(classifier, sess, (x_batch, y_batch))
        pred_test_list.append(metrics.f1_score(y_batch, pred_test, average='weighted'))
        attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
        attn_list.append(attn)

        for ele in x_batch:
            sentence = []
            for idx in ele:
                if idx != 0.0:
                    sentence.append(index_to_word[idx])
                else:
                    sentence.append("<pad>")
            test_sentence_list.append(sentence)

        test_result_list.append(pred_test)

        test_true_list.append(y_batch)
        test_acc += acc
        cnt += 1

    test_sentence_list_v = np.vstack(test_sentence_list)
    test_result_list_v = np.vstack(test_result_list)
    test_true_list_v = np.vstack(test_true_list)
    test_result_list_v_r = np.reshape(test_result_list_v, [-1, 1])
    test_true_list_v_r = np.reshape(test_true_list_v, [-1, 1])
    attn_list_v = np.vstack(attn_list)

    crosstab_y = np.squeeze(test_true_list_v_r)
    crosstab_y_pred = np.squeeze(test_result_list_v_r)
    pd.crosstab(crosstab_y, crosstab_y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    np.save('test_sentence_list.npy', test_sentence_list_v)
    np.save('test_result_list.npy', test_result_list_v_r)
    np.save('test_true_list.npy', test_true_list_v_r)
    np.save('attn_list.npy', attn_list_v)
    # attn_dic['sentece'] = np.vstack(test_sentence_list)
    # attn_dic['pre_label'] = np.vstack(test_result_list)
    # attn_dic['true_label'] = np.vstack(test_true_list)
    # attn_dic['attn'] = np.vstack(attn_list)
    # json.dump(attn_dic, open('../result/attn_dic.json', 'w'))
    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
    print("Test f1_score : %f %%" % (float(np.sum(np.asarray(pred_test_list))) / len(pred_test_list)))
    x_cor = []
    for i in range(config["train_epoch"]):
        x_cor.append(i + 1)

    plt.figure(figsize=(6, 6), dpi=80)
    # 创建第一个画板
    plt.figure(1)
    # 将第一个画板划分为2行1列组成的区块，并获取到第一块区域
    ax1 = plt.subplot(211)

    # 在第一个子区域中绘图
    plt.plot(np.asarray(x_cor, dtype=int), np.asarray(total_train_loss), 'r')
    plt.plot(np.asarray(x_cor, dtype=int), np.asarray(total_eval_loss), 'g')
    # 选中第二个子区域，并绘图
    ax2 = plt.subplot(212)
    plt.plot(np.asarray(x_cor, dtype=int), np.asarray(f1_train_list), 'r')
    plt.plot(np.asarray(x_cor, dtype=int), np.asarray(f1_eval_list), 'g')
    #
    # ax3 = plt.subplot(313)
    # plot_confusion_matrix(crosstab_y, crosstab_y_pred)
    plt.show()
