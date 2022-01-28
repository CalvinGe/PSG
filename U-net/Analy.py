import pandas as pd
import numpy as np


# conf_matrix = np.loadtxt('./sleep_stages_conf_mat.txt')
#
# print("conf_matrix", conf_matrix)
# TP = np.zeros((5,))
# FP = np.zeros((5,))
# TN = np.zeros((5,))
# FN = np.zeros((5,))
# SUM = np.sum(conf_matrix)
# for i in range(5):
#     TP[i] = conf_matrix[i, i]
#     FP[i] = np.sum(conf_matrix, axis=1)[i] - TP[i]
#     TN[i] = SUM + TP[i] - np.sum(conf_matrix, axis=1)[i] - np.sum(conf_matrix, axis=0)[i]
#     FN[i] = np.sum(conf_matrix, axis=0)[i] - TP[i]
# accuracy = (TP + TN) / SUM
# specificity = TN / (TN + FP)
# sensitivity = TP / (TP + FN)
# print("accuracy: ", accuracy)
# print("specificity: ", specificity)
# print("sensitivity: ", sensitivity)


conf_matrix = np.loadtxt('./arousal_conf_mat.txt')

print("conf_matrix", conf_matrix)
TN = conf_matrix[0, 0]
FN = conf_matrix[0, 1]
TP = conf_matrix[1, 1]
FP = conf_matrix[1, 0]
SUM = np.sum(conf_matrix)

accuracy = (TP + TN) / SUM
specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)
print("accuracy: ", accuracy)
print("specificity: ", specificity)
print("sensitivity: ", sensitivity)