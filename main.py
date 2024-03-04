import torch
import torch.nn as nn
import jieba
from torchtext.legacy import data
from torchtext.vocab import Vectors
import time
jieba.setLogLevel(jieba.logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 分词
def tokenizer(text):
    return [word for word in jieba.lcut(text) if word not in stop_words]
# 去停用词
def get_stop_words():
    file_object = open('D:\pythonProject5\Data\stopword.txt', encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

stop_words = get_stop_words()  # 加载停用词表
text = data.Field(sequential=True,
                  lower=True,
                  tokenize=tokenizer,
                  stop_words=stop_words)
label = data.Field(sequential=False)

train, val = data.TabularDataset.splits(
    path='D:\pythonproject5\Data\\',
    skip_header=True,
    train='data_train.csv',
    validation='data_val.csv',
    format='csv',
    fields=[('label', label), ('text', text)],
)

#print(train[0].text)

text.build_vocab(train, val, vectors=Vectors(name='D:\pythonProject5\.vector_cache\glove.6B.200d.txt'))
label.build_vocab(train, val)

# print(text.vocab.freqs)
# print(text.vocab.vectors)
# print(text.vocab.vectors.size())


embedding_dim = text.vocab.vectors.size()[-1]
vectors = text.vocab.vectors
batch_size = 128

train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(val)),# 训练集设置batch_size,验证集整个集合用于测试
    )

vocab_size = len(text.vocab)
label_num = len(label.vocab)

#print('len(TEXT.vocab)', len(text.vocab))
#print('TEXT.vocab.vectors.size()', text.vocab.vectors.size())

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            # batch_first=True,
            bidirectional=True
        )
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len], LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置后再提取词特征
        # 输出形状 outputs: [seq_len, batch_size, embedding_dim]
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings, 因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(seq_len, batch_size, 2*num_hiddens)
        outputs, _ = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。
        # 它的形状为 : [batch_size, 4 * num_hiddens]
        encoding = torch.cat((outputs[0], outputs[-1]), dim=-1)
        outs = self.decoder(encoding)
        return outs

embedding_dim, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab_size, embedding_dim, num_hiddens, num_layers)
#print(net)

lr, num_epochs = 0.001, 30
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

if __name__ == "__main__":
    def train(train_iter, val_iter, net, loss, optimizer, num_epochs):
        batch_count = 0
        best_acc = 0  # 2 初始化best test accuracy
        with open("acc.txt", "w") as f:
            for epoch in range(num_epochs):
                train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
                for batch_idx, batch in enumerate(train_iter):
                    X, y = batch.text, batch.label
                    X = X.permute(1, 0)
                    y.data.sub_(1)  #X转置 y下标从0开始
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    train_l_sum += l.item()
                    train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                    n += y.shape[0]
                    batch_count += 1
                val_acc = evaluate_accuracy(val_iter, net)
                print(
                    'epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
                    % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                       val_acc, time.time() - start))
                # 将每次测试结果实时写入acc.txt文件中
                f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, val_acc))
                f.write('\n')
                f.flush()
                # 记录最佳测试分类准确率并写入best_acc.txt文件中
                if val_acc > best_acc:
                    f3 = open("best_acc.txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, val_acc))
                    f3.close()
                    best_acc = val_acc
                    torch.save(net, 'D:\pythonProject5\\net.pth')


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.text, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)  #X转置 y下标从0开始
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train() # 改回训练模式
            else: # 自定义的模型
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

train(train_iter, val_iter, net, loss, optimizer, num_epochs)


