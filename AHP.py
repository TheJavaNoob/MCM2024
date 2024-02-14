import numpy as np

array = np.array([[1, 1/5, 1/10], 
              [5, 1, 1/5], 
              [10, 5, 1]])

n = array.shape[0]
RI_list = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58,
                1.59]
# 矩阵的特征值和特征向量
eig_val, eig_vector = np.linalg.eig(array)
# 矩阵的最大特征值
max_eig_val = np.max(eig_val)
# 矩阵最大特征值对应的特征向量
max_eig_vector = eig_vector[:, np.argmax(eig_val)].real
# 矩阵的一致性指标CI
CI_val = (max_eig_val - n) / (n - 1)
# 矩阵的一致性比例CR
CR_val = CI_val / (RI_list[n - 1])

weight = max_eig_vector / np.sum(max_eig_vector)

weight /= weight[0]
print("CI:", CI_val, "CR:", CR_val)
print("Weight:", weight)
    
