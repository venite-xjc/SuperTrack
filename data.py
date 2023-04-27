import torch
import torch.nn as nn
from torch.utils.data import Dataset
from bvh import Bvh
import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

class BVHdataset(Dataset):
    def __init__(self, root = "./BVH/jumps1_subject5.bvh", window_size = 32):
        super(BVHdataset).__init__()
        self.window_size = window_size
        self.bvh = Bvh()

        self.bvh.parse_file(root)
        self.bvh.all_frame_poses()
        joint_names = self.bvh.joint_names()
        mask = [i for i in range(len(joint_names)) if 'end' not in joint_names[i]]

        self.world_space_p = self.bvh.world_space_p[:, mask, :] #global position
        self.world_space_r = self.bvh.world_space_r[:, mask, :] #global rotation
        self.world_space_vp = self.bvh.world_space_vp[:, mask, :] #global velocity
        self.world_space_vr = self.bvh.world_space_vr[:, mask, :] #global rotational velocity

        mask = mask[1:]

        self.pd_r = self.bvh.pd_r[:, mask, :]
        self.pd_vr = self.bvh.pd_vr[:, mask, :]

        self.root_position = self.world_space_p[:, 0, :]
        self.root_rotation = []
        for i in range(self.world_space_r.shape[0]):
            self.root_rotation.append(quat2mat(self.world_space_r[i, 0, :]))
        self.root_rotation = np.array(self.root_rotation)

        self.local_position = np.zeros_like(self.world_space_p)
        self.local_volecity = np.zeros_like(self.world_space_vp)
        self.local_rotation = np.zeros([self.local_position.shape[0], self.local_position.shape[1], 6])
        self.local_rvelocity = np.zeros_like(self.world_space_vr)
        self.height = np.zeros([self.local_position.shape[0], self.local_position.shape[1], 1])
        self.up_vector = np.zeros([self.local_position.shape[0], 3])
        for frame in range(self.local_position.shape[0]):
            for body in range(self.local_position.shape[1]):
                self.local_position[frame, body, :] = self.root_rotation[frame, :].T @ (self.world_space_p[frame, body, :]-self.root_position[frame, :])
                self.local_volecity[frame, body, :] = self.root_rotation[frame, :].T @ (self.world_space_vp[frame, body, :])
                self.local_rotation[frame, body, :] = (self.root_rotation[frame, :].T @ (euler2mat(*np.deg2rad(self.world_space_r[frame, body, :]))))[:, :2].reshape(6)
                self.local_rvelocity[frame, body, :] = self.root_rotation[frame, :].T @ self.world_space_vr[frame, body, :]
                self.height[frame, body, 0] = self.world_space_p[frame, body, 1]# y axis is upforward
                self.up_vector[frame, :] = self.root_rotation[frame, :].T @ np.array([0, 1, 0])# y axis is upforward

        print(self.local_position.shape)
        print(self.local_rvelocity.shape)
        print(self.local_rotation.shape)
        print(self.local_rvelocity.shape)
        print(self.height.shape)
        print(self.up_vector.shape)



    def __len__(self):
        return self.bvh.frames-self.window_size-1
    
    def __getitem__(self, index):
        frames = np.array([index+i for i in range(self.window_size+1)]) #p0, p1, p2,..., pn
        x_p = self.world_space_p[frames, :, :]
        x_rootp = self.root_position[frames, :]
        x_rootr = self.root_rotation[frames, :, :]


if __name__ == "__main__":
    a = BVHdataset()
    