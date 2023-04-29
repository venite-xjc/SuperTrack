import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation

a = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
print(np.rad2deg(mat2euler(a)))

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle

# 定义旋转向量
rot_vec = np.array([1, 0, 0])

# 构建旋转矩阵
theta = np.pi/4  # 旋转角度
rot_matrix = axangle2mat(rot_vec, theta)

# 将旋转矩阵转换为旋转向量
axis, angle = mat2axangle(rot_matrix)

print('旋转轴: ', axis)
print('旋转角度: ', np.rad2deg(angle))

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle

# 定义两个旋转向量
rot_vec1 = np.array([1, 0, 0])  # 旋转45度
rot_vec2 = np.array([0, 1, 0])  # 旋转60度

# 将旋转向量转化为旋转矩阵
rot_mat1 = axangle2mat(rot_vec1, np.pi / 4)
rot_mat2 = axangle2mat(rot_vec2, np.pi / 3)

# 定义一个向量
vec = np.array([[1, 2, 3]]).T

# 对向量进行旋转
rot_vec_sum = rot_mat2@rot_mat1

print("旋转向量相加后的结果：", rot_vec_sum)
print(mat2axangle(rot_vec_sum))

a = np.zeros([10,10, 6])
print(a, a.shape)

a = np.array([2, 2, 3])
mat = euler2mat(*a)

print(np.linalg.inv(mat))
print(mat.T)

mat = np.array([
    [1, 0, 1],
    [1, 2, 3],
    [0, 2, 1]
])

v = np.array([1, 2, 3])
print(mat@v)

def a():
    return 1, 2, 3, 4, 5, 6, 7

b = a()
for i in b:
    print(i)