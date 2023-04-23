from bvh import Bvh
import os
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root = "./BVH"
file_list = os.listdir(root)
file_list = [os.path.join(root, i) for i in file_list]

file = file_list[0]


class BVHLoader():
    def __init__(self, filename):
        self.filename = filename

    def load_file(self):
        with open(self.filename) as f:
            mocap = Bvh(f.read())

        self.name_list = mocap.get_joints_names()

        self.parent = {}
        for i in self.name_list:
            self.parent[i] = mocap.get_joint(i).parent.value[-1] if mocap.get_joint(i).parent.value!=[] else "None"

        self.offset = {}
        for i in self.name_list:
            self.offset[i] = mocap.joint_offset(i)

        self.data = [[float(element) for element in row] for row in mocap.frames]
        self.data = np.array(self.data)

        self.channels = []
        for i in self.name_list:
            self.channels += [i+'_'+j for j in mocap.joint_channels(i)]

        self.root = self.name_list[0]

    def get_rotation(self, name):
        index = [self.channels.index(name+"_Zrotation"), self.channels.index(name+"_Xrotation"), self.channels.index(name+"_Xrotation")]
        return self.data[:, index]
    

    
    @lru_cache(maxsize = 22)
    def get_position(self, name):
        if name == self.root:
            root_position_index = [self.channels.index(self.root+"_Xposition"), self.channels.index(self.root+"_Xposition"), self.channels.index(self.root+"_Xposition")]
            root_position = self.data[:, root_position_index]
            root_rotation_index = [self.channels.index(self.root+"_Zrotation"), self.channels.index(self.root+"_Xrotation"), self.channels.index(self.root+"_Yrotation")]
            root_rotation = self.data[:, root_rotation_index]
            root_rotation_R = np.array([self.get_composite_rotation(root_rotation[i, :]) for i in range(root_rotation.shape[0])])
            return root_position, root_rotation_R
        
        else:
            parent_pos, parent_R = self.get_position(self.parent[name])# get parent's position
            rotation = self.get_rotation(name)
            offset = np.array([self.offset[name]]).T
            delta_list = np.array([(i@offset).T for i in parent_R]).squeeze(1)

            R_list = [parent_R[i]@self.get_composite_rotation(rotation[i, :]) for i in range(rotation.shape[0])]

            return parent_pos+delta_list, R_list

    def get_composite_rotation(self, rotation):
        '''
        rotation: [angle_z, angle_x, angle_y]
        '''
        z = rotation[0]/180*np.pi
        y = rotation[1]/180*np.pi
        x = rotation[2]/180*np.pi
        Rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ])
        Ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])
        # R = Rx@Ry@Rz#extrinsic rotation
        R = Rz@Ry@Rx#intrinsic rotation

        return R

            
        
        

a = BVHLoader(file)
a.load_file()
points = []
for i in a.name_list:
    k, _ = a.get_position(i)
    points.append(k)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

print(points.__len__())
for i in range(10):
    for j in points:
        print(j[i, 0], j[i, 1], j[i, 2])
        ax.scatter(j[i, 0], j[i, 1], j[i, 2], c='r', marker='o')
plt.show()
# 显示图像
