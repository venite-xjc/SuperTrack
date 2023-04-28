import torch
import torch.nn as nn
import torch.optim as optim
from data import BVHdataset
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.axangles import axangle2mat, mat2axangle

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = 5, hidden_units = 1024, act_layer = nn.ELU):
        self.input_size = input_size

        layers = []
        layers+=[nn.Linear(input_size, hidden_units), act_layer()]
        for i in range(hidden_layers-1):
            layers+=[nn.Linear(hidden_units, hidden_units), act_layer()]
        layers+=[nn.Linear(hidden_units, output_size)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        assert x.shape[-1] == self.input_size
        x = self.layers(x)
        return x

class SuperTrack:
    def __init__(self):
        self.world = Net(100, 100)
        self.world_batchsize = 2048
        self.world_lr = 0.001
        self.world_window = 8
        self.world_optimizer = optim.RAdam(self.world.parameters(), lr=self.world_lr)

        self.policy = Net(100, 100)
        self.policy_batchsize = 1024
        self.policy_lr = 0.0001
        self.policy_window = 32
        self.policy_optimizer = optim.RAdam(self.policy.parameters(), lr=self.policy_lr)

        self.EPOCH = 10
        self.iterations = 10000

        self.train_dataset = BVHdataset()

        self.dtime = 1/30

    def local(self, pos, vel, rot, ang):
        '''
        convert world space input into local space

        pos:[B, body, 3] position in world space
        vel:[B, body, 3] velocity in world space
        rot:[B, body, 4] rotation in world space, quaternion
        ang:[B, body, 3] rotational velocity in world space, axis-angle
        '''
        
        BS, B, _ = pos.shape
        
        lpos = torch.empty((BS, B, 3))
        lvel = torch.empty((BS, B, 3))
        lrot = torch.empty((BS, B, 6))
        lang = torch.empty((BS, B, 3))
        height = torch.empty((BS, 1))
        lup = torch.empty((BS, 3))

        for i in range(BS):
            for j in range(B):
                inv = quat2mat(rot[i, 0, :]).T
                lpos[i, j, :] = inv @ (pos[i, j, :]-pos[i, 0, :])
                lvel[i, j, :] = inv @ (vel[i, j, :])
                lrot[i, j, :] = (inv @ quat2mat(rot[i, j, :]))[:, :2].reshape(6)
                lang[i, j, :] = inv @ ang[i, j, :]
                height[i, :] = pos[i, j, 1] #y axis is upforward
                lup = inv @ torch.tensor([0, 1, 0])
        
        return lpos, lvel, lrot, lang, height, lup
    
    

    def forward(self, pos, vel, rot, ang, j_rot, j_ang):
        '''
        pos: [B, frames+1, bodies, 3]
        vel: [B, frames+1, bodies, 3]
        rot: [B, frames+1, bodies, 4] quatenion
        ang: [B, frames+1, bodies, 3] axis-angle
        j_rot: [B, frames+1 joints, 4] quatenion
        j_ang: [B, frames+1, joints, 3] axis_angle
        '''

        # self.world.train()
        # self.policy.eval() # not update policy
        # for param in self.policy.parameters():
        #     param.requires_grad =False

        #initial state, S0<-P0
        s_pos = pos[:, 0, :, :] # [B, body, dim]
        s_vel = vel[:, 0, :, :]
        s_rot = rot[:, 0, :, :]
        s_ang = ang[:, 0, :, :]

        for i in range(self.world_window):
            #produce o:[B, j, 3] represented as axis-angle
            o = self.policy(self.local(s_pos, s_vel, s_rot, s_ang), self.local(pos[:, i+1, :, :], vel[:, i+1, :, :], rot[:, i+1, :, :], ang[:, i+1, :, :]))
            o = o+0.1*torch.randn_like(o) #add noise

            # get PD target
            T_rot = j_rot[:, i+1, :, :] #[B, joints, 4]
            for i in range(T_rot.shape[0]):
                for j in range(T_rot.shape[1]):
                    T_rot[i, j, :] = mat2quat(axangle2mat(o[i, j, :]) @ quat2mat(T_rot[i, j, :]))
            T_ang = j_ang[:, i+1, :, :]
            
            # get accelerations, both shapes are [B, bodies, 3]
            position_acceleration, rotation_acceleration = self.world(self.local(s_pos, s_vel, s_rot, s_ang), T_rot, T_ang)

            # turn accelerations into world space 
            for i in range(position_acceleration.shape[0]):
                rootr = s_rot[i, 0, :] #quat
                rootr_matrix = quat2mat(rootr)
                for j in position_acceleration.shape[1]:
                    position_acceleration[i, j, :] = rootr_matrix @ position_acceleration[i, j, :]
                    rotation_acceleration[i, j, :] = rootr_matrix @ rotation_acceleration[i, j, :]

            #update
            s_vel = s_vel + self.dtime*position_acceleration
            s_ang = s_ang + self.dtime*rotation_acceleration
            s_pos = s_vel + self.dtime*s_vel
            for i in range(s_rot.shape[0]):
                for j in range(s_rot.shape[1]):
                    s_pos = mat2quat(axangle2mat(self.dtime*s_ang[i, j, :]) @ quat2mat(s_rot[i, j, :]))


            
    def train_world_loss(self, state_gt, state_pred):
        w_pos = w_vel = w_rot = w_ang = 1
        L1loss = torch.nn.L1Loss()
        for i in range(len(state_pred)):
            if i == 0:
                loss = w_pos*L1loss()

    def train_policy_loss(self):
        pass

    def train(self):
        for epoch in self.EPOCH:
            for pos, vel, rot, ang, j_rot, j_ang in self.train_dataset:
                self.train_world(pos, vel, rot, ang)

                self.train_policy(pos, vel, rot, ang)

