import numpy as np
import scipy.io
import cv2
from mne.io import concatenate_raws, read_raw_edf


def import_hospital(edf_file):
    raw = read_raw_edf(edf_file, preload=False)
    # print(raw.info)        # sfreq=512
    raw = raw.resample(256)

    image_data = raw.get_data()
    image_data = image_data[:13, :]
    # print(image_data.shape)  # (13, 19595776)
    return image_data


def anchor(ref, ori):  # input m*n np array
    d0 = ori.shape[0]
    d1 = ori.shape[1]
    ref = cv2.resize(ref, (d1, d0), interpolation=cv2.INTER_AREA)
    ori_new = ori.copy()
    for i in range(d0):
        ori_new[i, np.argsort(ori[i, :])] = ref[i, :]
    return ori_new


ref1024 = np.load('./data/ref1024_13c.npy')
path1 = '../data/医院数据/EDF/'
path2 = './data/avg16_16m_multi_task_image/'

size = 4096 * 256
num_pool = 4
scale_pool = 2 ** num_pool
num = 0

for num in range(15):
    the_id = '00' + str(num + 1)
    the_id = the_id[-3:]
    image_ori = import_hospital(path1 + the_id + '.edf')
    image = anchor(ref1024, image_ori)
    d0 = image.shape[0]
    d1 = image.shape[1]
    if d1 < size * scale_pool:
        image = np.concatenate((image, np.zeros((d0, size * scale_pool - d1))), axis=1)
    image = cv2.resize(image, (size, d0), interpolation=cv2.INTER_AREA)
    print('image: ', image.shape)

    np.save(path2 + the_id, image)
