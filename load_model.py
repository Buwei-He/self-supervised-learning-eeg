import numpy as np
a = np.load('./Series2Vec/Results/Pre_Training/EEG/AC_0.77/k_fold_1/analysis_test_epoch_1.npy', allow_pickle=True).item()
print(a.keys())