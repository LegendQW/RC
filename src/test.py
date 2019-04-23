# coding=utf-8


from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from utils.prepare_data import *
import time
from utils.model_helper import *
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib
import json

x_cor = []
for i in range(40):
    x_cor.append(i + 1)

plt.figure(figsize=(6, 6), dpi=80)
# 创建第一个画板
plt.figure(1)
# 将第一个画板划分为2行1列组成的区块，并获取到第一块区域
ax1 = plt.subplot(211)

total_train_loss = [2.05534854465, 1.75621202257, 1.6364537557, 1.55046708849, 1.49197421604, 1.44693840875,
                    1.3980957879, 1.37551998562, 1.32999606662, 1.31203698052, 1.27717709012,
                    1.24556681315, 1.22207048204, 1.22058690389, 1.19327689277, 1.17926669651, 1.15783250597,
                    1.14712897407, 1.11980183919, 1.11987592909, 1.09216325548, 1.07999267578, 1.07136484782,
                    1.06470082601, 1.04534132216, 1.03427369859, 1.01529973348, 1.00383851793, 0.998642306858,
                    0.978637017144, 0.97811694675, 0.959238433838, 0.961452907986, 0.949496629503, 0.957493760851,
                    0.942202674018, 0.912179226345, 0.90288204617, 0.903758070204, 0.895220862495]
total_eval_loss = [1.77053153515, 1.65716898441, 1.58311069012, 1.51768326759, 1.49724376202, 1.39404726028,
                   1.36095058918, 1.3737206459, 1.27591228485, 1.30811822414, 1.23204278946,
                   1.25399935246, 1.24137508869, 1.25316643715, 1.19640564919, 1.1621773243, 1.16798269749,
                   1.17382967472, 1.19637060165, 1.09977006912, 1.15461552143, 1.1444889307, 1.16820299625,
                   1.10369706154, 1.11606943607, 1.17465388775, 1.11257219315, 1.11971771717, 1.08676719666,
                   1.1048463583, 1.12983572483, 1.10184192657, 1.06891465187, 1.12589335442, 1.12303245068,
                   1.08565580845, 1.13885223866, 1.13767242432, 1.21622788906, 1.16482293606]


# 在第一个子区域中绘图
# plt.plot(np.asarray(x_cor, dtype=int), np.asarray(total_train_loss), 'r')
# plt.plot(np.asarray(x_cor, dtype=int), np.asarray(total_eval_loss), 'g')
# plt.show()


def plot(mat, xlen, ylen, xlabel, ylabel, rotate=True):
    if rotate:
        mat = mat.T
        xlen, ylen = ylen, xlen
        xlabel, ylabel = ylabel, xlabel
    print (mat[:ylen, :xlen])
    plt.pcolor(mat[:ylen, :xlen], cmap=plt.cm.gray_r)
    _ = plt.xticks(np.linspace(0.5, xlen - 0.5, xlen), xlabel)
    _ = plt.yticks(np.linspace(0.5, ylen - 0.5, ylen), ylabel)


def get_len(attn_list):
    length = 0
    for ele in attn_list:
        if ele > 0:
            length = length + 1
    return length


def attention_plot(attn_matrix, index, id2r):
    attn_matrix_spe = attn_matrix[index:]
    atten_len = get_len(attn_matrix_spe[index])
    xlabel = sentence_list[index][:atten_len]
    ylabel = []
    for ele in pred_result[index]:
        ylabel.append(id2r[ele])
    plot(attn_matrix_spe.T, atten_len, 1, xlabel=xlabel, ylabel=ylabel, rotate=True)
    plt.show()


with open('../data/r2id.json', 'r') as f_r2id:
    # train = json.load(f_train)
    r2id = json.load(f_r2id)

id2r = {v: k for k, v in r2id.items()}
plt.figure(figsize=(6, 6))
attn_matrix = np.load("attn_list.npy")

# print (attn_matrix[0][11])
# print (attn_matrix[0][6])
# temp = attn_matrix[0][11]

# attn_matrix[0][11] = attn_matrix[0][6]
# print (attn_matrix[0][11])
# attn_matrix[0][6] = temp
# print (attn_matrix[0][6])
# sentence_list = np.load("test_sentence_list.npy")
# pred_result = np.load("test_result_list.npy")
# attention_plot(attn_matrix, 100, id2r)

confuse_matrix = np.array([[293, 1, 0, 1, 5, 0, 0, 3, 5, 12],
                           [1, 248, 4, 2, 1, 7, 12, 3, 8, 16],
                           [0, 2, 166, 5, 1, 1, 1, 3, 0, 12],
                           [0, 3, 6, 254, 0, 1, 0, 3, 2, 23],
                           [9, 3, 3, 2, 199, 1, 0, 1, 9, 31],
                           [1, 8, 1, 1, 0, 109, 0, 3, 8, 24],
                           [0, 12, 1, 1, 1, 1, 196, 1, 2, 13],
                           [1, 6, 0, 2, 2, 1, 1, 223, 5, 14],
                           [5, 4, 2, 1, 5, 9, 1, 5, 179, 18],
                           [22, 49, 24, 27, 18, 24, 34, 36, 34, 186]], dtype=np.int32)
x_labels = ['C-E', 'C-W', 'C-C', 'E-D', 'E-O', 'I-A', 'M-C', 'M-T', 'P-P', "_O_"]
plot(confuse_matrix, 10, 10, xlabel=x_labels, ylabel=x_labels, rotate=False)
plt.show()
# print(sentence_list[0])
