import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from bvh import Bvh
import numpy as np
import pytorch3d.transforms as pyt
import os
import multiprocessing

def load_file(file_name, window_size, world_space_p, world_space_r, world_space_vp, world_space_vr, pd_r, pd_vr, hash, lock):
    bvh = Bvh()
    bvh.parse_file(file_name)
    bvh.all_frame_poses()
    joint_names = bvh.joint_names()
    mask = [i for i in range(len(joint_names)) if 'end' not in joint_names[i]]
    
    lock.acquire()
    world_space_p.append(bvh.world_space_p[:, mask, :]) #global position
    world_space_r.append(bvh.world_space_r[:, mask, :]) #global rotation
    world_space_vp.append(bvh.world_space_vp[:, mask, :]) #global velocity
    world_space_vr.append(bvh.world_space_vr[:, mask, :]) #global rotational velocity

    mask = mask[1:]

    pd_r.append(bvh.pd_r[:, mask, :])
    pd_vr.append(bvh.pd_vr[:, mask, :])
    hash+=[[len(pd_r)-1, i] for i in range(bvh.pd_r.shape[0]-window_size-1)]
    lock.release()

    print("finish ", file_name)

class BVHdataset(Dataset):
    def __init__(self, window_size = 32):
        super(BVHdataset).__init__()
        self.window_size = window_size
        
        self.world_space_p = multiprocessing.Manager().list()
        self.world_space_r = multiprocessing.Manager().list()
        self.world_space_vp = multiprocessing.Manager().list()
        self.world_space_vr = multiprocessing.Manager().list()
        self.pd_r = multiprocessing.Manager().list()
        self.pd_vr = multiprocessing.Manager().list()
        self.hash = multiprocessing.Manager().list()
        lock = multiprocessing.Lock()

        recprdprocess=[]
        root_list = os.listdir("./BVH")
        for root in root_list[:-10]:
            p = multiprocessing.Process(target=load_file, args=(os.path.join('./BVH', root), window_size, self.world_space_p, self.world_space_r, self.world_space_vp, \
                                                                self.world_space_vr, self.pd_r, self.pd_vr, self.hash, lock))
            p.start()
            recprdprocess.append(p)

        for p in recprdprocess:
            p.join()

        self.world_space_p = list(self.world_space_p)
        self.world_space_r = list(self.world_space_r)
        self.world_space_vp = list(self.world_space_vp)
        self.world_space_vr = list(self.world_space_vr)
        self.pd_r = list(self.pd_r)
        self.pd_vr = list(self.pd_vr)
        self.hash = list(self.hash)
        # for i in range(len(self.world_space_p)):
        #     print(self.world_space_p[i].shape, self.world_space_r[i].shape, self.world_space_vp[i].shape, self.world_space_vr[i].shape, self.pd_r[i].shape, self.pd_vr[i].shape)

            # del bvh
            # bvh = Bvh()
            # bvh.parse_file(os.path.join("./BVH", root))
            # bvh.all_frame_poses()
            # joint_names = bvh.joint_names()
            # mask = [i for i in range(len(joint_names)) if 'end' not in joint_names[i]]


            # self.world_space_p.append(bvh.world_space_p[:, mask, :]) #global position
            # self.world_space_r.append(bvh.world_space_r[:, mask, :]) #global rotation
            # self.world_space_vp.append(bvh.world_space_vp[:, mask, :]) #global velocity
            # self.world_space_vr.append(bvh.world_space_vr[:, mask, :]) #global rotational velocity

            # # print(pyt.quaternion_to_matrix(self.world_space_r[0, :, :]))
            # # print(pyt.quaternion_to_matrix(self.world_space_r[1, :, :]))
            # # print(torch.matmul(pyt.axis_angle_to_matrix(self.world_space_vr[0, :, :]/30),pyt.quaternion_to_matrix(self.world_space_r[0, :, :])))
            # # print(torch.matmul(pyt.quaternion_to_matrix(self.world_space_r[0, :, :]),pyt.axis_angle_to_matrix(self.world_space_vr[0, :, :]/30)))
            # # raise Exception

            # mask = mask[1:]

            # self.pd_r.append(bvh.pd_r[:, mask, :])
            # self.pd_vr.append(bvh.pd_vr[:, mask, :])
            # self.hash+=[[file, i] for i in range(bvh.pd_r.shape[0]-window_size-1)]

            # file+=1
            



    def __len__(self):
        return len(self.hash)
    
    def __getitem__(self, index):
        frames = np.array([self.hash[index][1]+i for i in range(self.window_size+1)]) #p0, p1, p2,..., pn
        
        return self.world_space_p[self.hash[index][0]][frames, :, :], \
                self.world_space_vp[self.hash[index][0]][frames, :, :], \
                self.world_space_r[self.hash[index][0]][frames, :, :], \
                self.world_space_vr[self.hash[index][0]][frames, :, :], \
                self.pd_r[self.hash[index][0]][frames, :, :], \
                self.pd_vr[self.hash[index][0]][frames, :, :]


    def get_loader(self, batch_size, num_worker = 4, shuffle = False):
        loader = DataLoader(dataset = self, batch_size=batch_size, num_workers=num_worker, shuffle=shuffle, drop_last=True)
        return loader

if __name__ == "__main__":
    a = BVHdataset()
    # print(a[0])
    for i in a[0]:
        print(i, i.shape)