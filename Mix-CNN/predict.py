import myFunc
import Models
import Forward
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_file = pd.read_table('test_data.txt', header=None)
anno_file = pd.read_table('test_anno.txt', header=None)
BATCH_SIZE = 80
SEQ_LEN = 5
N_chn = 3


x_test, y_test = myFunc.extract(data_file.iloc[0, 0], anno_file.iloc[0, 0])
for i in range(1, data_file.shape[0]):
    x_tmp, y_tmp = myFunc.extract(data_file.iloc[i, 0], anno_file.iloc[i, 0])
    x_test = np.concatenate((x_test, x_tmp), axis=0)
    y_test = np.concatenate((y_test, y_tmp), axis=0)


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


test_data = EEGdataset(x_test, y_test)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE * SEQ_LEN, shuffle=True)

checkpoint = torch.load('./models_param/pretrain-mix.pth')
part1_attention = Models.attention(BATCH_SIZE, SEQ_LEN, device).to(device)
part1_cnn = Models.part1(BATCH_SIZE, SEQ_LEN).to(device)
part4 = Models.part4(N_chn).to(device)

part1_attention.load_state_dict(checkpoint['part1_attention'])
part1_cnn.load_state_dict(checkpoint['part1_cnn'])
part4.load_state_dict(checkpoint['part4'])

with torch.no_grad():
    conf_matrix = np.zeros((5, 5))
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        chn = []
        for i in range(3):
            t1 = part1_attention(inputs[:, i, :])
            t2 = part1_cnn(inputs[:, i, :])
            chn.append(t1 + t2)
        outputs = torch.cat((chn[0], chn[1], chn[2]), dim=1)
        outputs = part4(outputs)

        _, prediction = outputs.max(dim=1)
        for p, t in zip(prediction, labels):
            conf_matrix[p, t] += 1
    TP = np.zeros((5,))
    FP = np.zeros((5,))
    TN = np.zeros((5,))
    FN = np.zeros((5,))
    SUM = np.sum(conf_matrix)
    for i in range(5):
        TP[i] = conf_matrix[i, i]
        FP[i] = np.sum(conf_matrix, axis=1)[i] - TP[i]
        TN[i] = SUM + TP[i] - np.sum(conf_matrix, axis=1)[i] - np.sum(conf_matrix, axis=0)[i]
        FN[i] = np.sum(conf_matrix, axis=0)[i] - TP[i]
    accuracy = (TP + TN) / SUM
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print("accuracy: ", accuracy)
    print("specificity: ", specificity)
    print("sensitivity: ", sensitivity)


