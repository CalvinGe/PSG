import numpy as np
import pandas as pd
import pickle
import wfdb
import scipy.io
from mne.io import concatenate_raws, read_raw_edf

num_class = 8
scale = 16
set_sleep = {'W', 'N1', 'N2', 'N3', 'R'}
# U-0; W-1, N1-2; N2-3; N3-4; R-5; arousal-6; apnea-7

for num in range(15):
    the_id = '00' + str(num + 1)
    the_id = the_id[-3:]

    label_ori = np.zeros((num_class, 2048 * 512))
    label_ori[0, :] = 1  # undefined as default

    path1 = './data/apnea/'
    path2 = './data/new_arousal/'
    path3 = '../data/医院数据/STAGE/'
    path4 = './data/avg16_16m_multi_task_image/'
    path5 = '../data/医院数据/EDF/'

    raw = read_raw_edf(path5 + the_id + '.edf', preload=False)
    # print(raw.info)        # sfreq=512

    image = raw.get_data()
    d1 = image.shape[1] // scale

    sleep_anno = pd.read_table(path3 + the_id + '.TXT', header=None)
    for i in range(len(sleep_anno)):
        start = i * 30 * 256 // scale
        if i == len(sleep_anno) - 1:
            end = d1
        else:
            end = (i+1) * 30 * 256 // scale
        if sleep_anno.iloc[i, 0] == 'W':
            cat = 1
        elif sleep_anno.iloc[i, 0] == 'N1':
            cat = 2
        elif sleep_anno.iloc[i, 0] == 'N2':
            cat = 3
        elif sleep_anno.iloc[i, 0] == 'N3':
            cat = 4
        elif sleep_anno.iloc[i, 0] == 'R':
            cat = 5
        label_ori[cat, start:end] = 1
        label_ori[0, start:end] = 0

    arousal_anno = np.loadtxt(path2 + the_id + '_arousal.txt')
    for i in range(len(arousal_anno)):
        start = arousal_anno[i, 0] * 256 // scale
        start = int(start)
        end = (arousal_anno[i, 0] + arousal_anno[i, 1]) * 256 // scale
        end = int(end)
        label_ori[6, start:end] = 1

    apnea_anno = np.loadtxt(path1 + the_id + '_apnea.txt')
    for i in range(len(apnea_anno)):
        start = arousal_anno[i, 0] * 256 // scale
        start = int(start)
        end = (arousal_anno[i, 0] + arousal_anno[i, 1]) * 256 // scale
        end = int(end)
        label_ori[7, start:end] = 1

    label_ori[:, d1:] = -1  # masked
    print(d1)
    print('label: ', label_ori.shape)  # (8, 2048*512)

    np.save('./data/avg16_16m_multi_task_label/' + the_id + '.npy', label_ori)