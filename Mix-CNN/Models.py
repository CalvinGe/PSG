import mne
import math
import numpy as np
import torch
from torch import nn


# ========================= part1 -- part4  =============================

# part1: 效果非常好的一个CNN模块
# input.shape = epochs x 3000
# output.shape = epochs x 4480
class part1(nn.Module):
    def __init__(self, BATCH_SIZE, SEQ_LEN):
        super(part1, self).__init__()
        self.batch_size = BATCH_SIZE * SEQ_LEN
        self.DP = nn.Dropout(0.5)
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, padding=(10,),
                                            kernel_size=(50,), stride=(6,)),
                                  nn.MaxPool1d(kernel_size=8, stride=8),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Conv1d(64, 128, (8,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (8,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.MaxPool1d(4, 4))
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, padding=(10,),
                                            kernel_size=(200,), stride=(16,)),
                                  nn.MaxPool1d(kernel_size=6, stride=6),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Conv1d(64, 128, (32,), (1,), padding=(16,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (8,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (8,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.MaxPool1d(4, 4))
        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, padding=(10,),
                                            kernel_size=(400,), stride=(50,)),
                                  nn.MaxPool1d(kernel_size=4, stride=4),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Conv1d(64, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(2,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(1,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (4,), (1,), padding=(1,)), nn.ReLU(),
                                  nn.MaxPool1d(2, 2))

    def forward(self, x):
        x = x.view(self.batch_size, 1, -1)
        x1 = self.cnn1(x)  # x1的size为BATCH_SIZE*128*16
        # print(x1.shape)
        x2 = self.cnn2(x)  # x2的size为BATCH_SIZE*128*11
        # print(x2.shape)
        x3 = self.cnn3(x)  # x3的size为BATCH_SIZE*128*8
        # print(x3.shape)

        x1 = x1.view(self.batch_size, -1)
        x2 = x2.view(self.batch_size, -1)
        x3 = x3.view(self.batch_size, -1)

        x = torch.cat((x1, x2, x3), dim=1)

        return self.DP(x)


# part2: 微调时才用到的BiGRU模块(带有残差连接)
# input.shape = SEQ_LEN x 4480 (类比于NLP中: 词向量长度/input_size = 4480, 句子长度为参数SEQ_LEN)
# output.shape = SEQ_LEN x 1024
# part2提取的是帧(epochs)和帧之间的时序关联
class part2(nn.Module):
    def __init__(self, BATCH_SIZE, SEQ_LEN):
        super(part2, self).__init__()
        self.GRU1 = nn.GRU(input_size=4480, hidden_size=512,
                           num_layers=1, bidirectional=True)
        self.DP = nn.Dropout(0.5)
        self.GRU2 = nn.GRU(input_size=2 * 512, hidden_size=512,
                           num_layers=1, bidirectional=True)
        self.Linear = nn.Linear(4480, 1024)
        self.Relu = nn.ReLU()
        self.seq_len = SEQ_LEN
        self.batch_size = BATCH_SIZE

    def forward(self, x):
        x1 = x.view(self.seq_len, self.batch_size, -1)

        x1, _ = self.GRU1(x1)
        x1 = self.DP(x1)
        x1 = self.Relu(x1)
        x1, _ = self.GRU2(x1)
        x1 = self.DP(x1).view(self.seq_len * self.batch_size, -1)
        x1 = self.Relu(x1)

        x2 = self.Relu(self.Linear(x))

        return x1 + x2


# part3, part4都是用以输出预测的线性层
# part3为预训练设计
class part3(nn.Module):
    def __init__(self, N_chn):
        super(part3, self).__init__()
        self.DP = nn.Dropout(0.5)
        self.Linear = nn.Linear(1024 * N_chn, 5)

    def forward(self, x):
        x = self.Linear(self.DP(x))
        return x


# part4为微调设计
class part4(nn.Module):
    def __init__(self, N_chn):
        super(part4, self).__init__()
        self.Linear = nn.Linear(4480 * N_chn, 5)

    def forward(self, x):
        return self.Linear(x)


# ==============================attention, part2_LSTM ===================================================

# part1 在一定程度上可以由以下的 attention 模块取代，二者用法相似
# attention 模块中需要调用的SelfAttention具体实现见程序底部
# convert 为重叠提取输入，原input.shape = epochs x 3000 经部分重叠提取后转为input.shape = epochs x 4480
class attention(nn.Module):
    def __init__(self, BATCH_SIZE, SEQ_LEN, device):
        super(attention, self).__init__()
        self.attention = SelfAttention(2, 20, 20, 0.3)
        self.batch_size = BATCH_SIZE * SEQ_LEN
        self.device = device

    def forward(self, x):
        # print(x.shape)
        x = convert(x, 4480, self.device)
        # print(x.shape)
        x = x.reshape(self.batch_size, 224, -1)
        x = self.attention(x)
        x = x.reshape(self.batch_size, -1)
        return x


# 输入 attention 中的 SelfAttention 时各个epoch之间需要有部分堆叠
# 重叠提取后 input.shape 从 epochs x 3000 变为 epochs x 4480
def convert(x, target_size, device="cpu"):
    num_epo = x.shape[0]
    att = torch.FloatTensor(num_epo, target_size).to(device)
    x = x.reshape(-1)
    att[0, :] = x[0:4480]
    for i in range(1, num_epo - 1):
        start = int((i + 0.5) * 3000 - 0.5 * target_size)
        end = int((i + 0.5) * 3000 + 0.5 * target_size)
        att[i, :] = x[start:end]
    att[num_epo - 1, :] = x[x.shape[0] - 4480:x.shape[0]]
    return att


# part2_LSTM仅仅是将part2中的BiGRU换成了LSTM
# input.shape = SEQ_LEN x 4480 (类比于NLP中: 词向量长度/input_size = 4480, 句子长度为参数SEQ_LEN)
# output.shape = SEQ_LEN x 1024
class part2_LSTM(nn.Module):
    def __init__(self, BATCH_SIZE, SEQ_LEN):
        super(part2_LSTM, self).__init__()
        self.LSTM1 = nn.LSTM(input_size=4480, hidden_size=512,
                             num_layers=1, bidirectional=True)
        self.DP = nn.Dropout(0.5)
        self.LSTM2 = nn.LSTM(input_size=2 * 512, hidden_size=512,
                             num_layers=1, bidirectional=True)
        self.Linear = nn.Linear(4480, 1024)
        self.Relu = nn.ReLU()
        self.seq_len = SEQ_LEN
        self.batch_size = BATCH_SIZE

    def forward(self, x):
        x1 = x.view(self.seq_len, self.batch_size, -1)

        x1, _ = self.LSTM1(x1)
        x1 = self.DP(x1)
        x1 = self.Relu(x1)
        x1, _ = self.LSTM2(x1)
        x1 = self.DP(x1).view(self.seq_len * self.batch_size, -1)
        x1 = self.Relu(x1)

        x2 = self.Relu(self.Linear(x))

        return x1 + x2


# =======================================自注意力机制的具体实现===========================================
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        attention_probs_dropout_prob = 0.3
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
