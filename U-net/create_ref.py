import numpy as np
import cv2
from mne.io import concatenate_raws, read_raw_edf


def import_hospital(edf_file):
    raw = read_raw_edf(edf_file, preload=False)
    # print(raw.info)        # sfreq=512
    raw = raw.resample(256)
    image_data = raw.get_data()
    image_data = image_data[:13, :]
    print(image_data.shape)  # (13, 9797888)
    return image_data


size = 10240000
ref = np.zeros((13, size)).astype('float32')
# Ch_names:
# ['E1-M2', 'E2-M2', 'F3-M2', 'F4-M1',
# 'C4-M1', 'C3-M2', 'O1-M2', 'O2-M1', 'ECG', 'SpO2',
# 'Nasal Pressure', 'Therm', 'THORACIC']

path = '../data/医院数据/EDF/'

for num in range(15):
    the_id = '00' + str(num+1)
    the_id = the_id[-3:]
    image = import_hospital(path + the_id + '.edf')
    d0 = image.shape[0]
    image = cv2.resize(image, (size, d0), interpolation=cv2.INTER_AREA)
    ref = ref + image

ref = ref / float(15)
np.save('./data/' + 'ref1024_13c', ref)