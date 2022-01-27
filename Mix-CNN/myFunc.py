import mne
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 将所需要读取的 EDF文件 的文件名写在data_file.txt/anno_file.txt中
# 当 EDF文件 很多时，我们可以自由选择训练集和测试集
# 注意：data_file.txt中为 通道数据EDF 文件的文件名
#      anno_file.txt中为 标签数据EDF 文件的文件名
#      data_file.txt与anno_file.txt中同一行应为同一个记录
DATA_FILE = pd.read_table('data_file.txt')
ANNO_FILE = pd.read_table('anno_file.txt')


# extract函数返回：2个ndarray
#  x_data: 包含'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'三个通道, x.shape = epochs x 3 x 3000
#  y_data: 每一帧的标签, y.shape = epochs
def extract(data_file, anno_file):
    # read files:
    raw_data = mne.io.read_raw_edf(data_file)
    anno_data = mne.read_annotations(anno_file)
    # label the data:
    raw_data.set_annotations(anno_data, emit_warning=False)

    # Link digits with annotations:
    annotation_desc_2_event_id = {'Sleep stage W': 0,
                                  'Sleep stage 1': 1,
                                  'Sleep stage 2': 2,
                                  'Sleep stage 3': 3,
                                  'Sleep stage 4': 3,
                                  'Sleep stage R': 4}
    event_id = {'Sleep stage W': 0,
                'Sleep stage 1': 1,
                'Sleep stage 2': 2,
                'Sleep stage R': 4}

    events_data, _ = mne.events_from_annotations(
        raw_data, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    tmax = 30. - 1. / raw_data.info['sfreq']

    # Only some of the data have stage 3/4, we append it here:
    if np.any(np.unique(events_data[:, 2] == 3)):
        event_id['Sleep stage 3/4'] = 3

    epochs_data = mne.Epochs(raw=raw_data, events=events_data,
                             event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)

    x_data = epochs_data.get_data(picks=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']) * 1e6
    y_data = epochs_data.events[:, 2]

    sleepStart = np.min(np.argwhere(y_data != 0))
    sleepEnd = len(y_data) - np.min(np.argwhere(y_data[::-1] != 0))
    # x_data = x_data[sleepStart:sleepEnd, :, :]
    # y_data = y_data[sleepStart:sleepEnd]

    return x_data, y_data


def data_preparation(BATCH_SIZE, SEQ_LEN, num_train):
    data_file = pd.read_table('data_file.txt', header=None)
    anno_file = pd.read_table('anno_file.txt', header=None)
    x_train, y_train = extract(data_file.iloc[0, 0], anno_file.iloc[0, 0])
    for i in range(1, num_train):
        x_tmp, y_tmp = extract(data_file.iloc[i, 0], anno_file.iloc[i, 0])
        x_train = np.concatenate((x_train, x_tmp), axis=0)
        y_train = np.concatenate((y_train, y_tmp), axis=0)

    x_test, y_test = extract(data_file.iloc[num_train, 0], anno_file.iloc[num_train, 0])
    if num_train + 1 < len(data_file):
        for i in range(num_train + 1, len(data_file)):
            x_tmp, y_tmp = extract(data_file.iloc[i, 0], anno_file.iloc[i, 0])
            x_test = np.concatenate((x_test, x_tmp), axis=0)
            y_test = np.concatenate((y_test, y_tmp), axis=0)

    # x的形状: epochs x 3(channels) x 3000; ndarray
    # y的形状: epochs; ndarray

    class EEGdataset(Dataset):
        def __init__(self, x, y):
            self.batch_size = BATCH_SIZE * SEQ_LEN
            num_batch = int(x.shape[0] / self.batch_size)
            self.x_data = torch.from_numpy(x[:num_batch * self.batch_size, :, :])
            self.x_data = self.x_data.type(torch.FloatTensor)
            self.y_data = torch.LongTensor(y[:num_batch * self.batch_size])
            self.len = self.y_data.shape[0]

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.len

    train_data = EEGdataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE * SEQ_LEN, shuffle=True)
    test_data = EEGdataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE * SEQ_LEN, shuffle=True)

    return train_loader, test_loader


def visualize(epoch_list, loss_list, acc_list, test_loss_list, test_acc_list, picture_name):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(np.array(epoch_list), np.array(loss_list),
             label='Train_loss', marker='o', linestyle='dashed', markersize=5)
    ax2.plot(np.array(epoch_list), np.array(acc_list), label='Train_accuracy',
             marker='s', markersize=5)
    ax1.plot(np.array(epoch_list), np.array(test_loss_list),
             label='Test_loss', marker='o', linestyle='dashed', markersize=5)
    ax2.plot(np.array(epoch_list), np.array(test_acc_list), label='Test_Accuracy',
             marker='s', markersize=5)
    ax1.set_ylim(0, 0.03)
    ax2.set_ylim(0, 1)
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    fig.savefig('./pictures/'+picture_name)
