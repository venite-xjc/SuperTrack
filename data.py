import torch
import torch.nn as nn
from torch.utils.data import Dataset
from bvh import Bvh

class BVHdataset(Dataset):
    def __init__(self, root, window_size):
        super(BVHdataset).__init__()
        self.window_size = window_size
        self.bvh = Bvh()

        self.bvh.parse_file("./BVH/jumps1_subject5.bvh")
        p, r = self.bvh.all_frame_poses()

        self.world_space_p = self.bvh.world_space_p
        self.world_space_r = self.bvh.world_space_r
        self.world_space_vp = self.bvh.world_space_vp

    def __len__(self):
        return self.bvh.frames-self.window_size-1
    
    def __getitem__(self, index):
        pass