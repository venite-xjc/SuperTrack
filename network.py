import torch
import torch.nn as nn
import torch.optim as optim
from data import BVHdataset
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.axangles import axangle2mat, mat2axangle
from functools import lru_cache
import pytorch3d.transforms as pyt
from tensorboardX import SummaryWriter
import os
import pdb
import random

class Policy_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = 5, hidden_units = 1024, act_layer = nn.ELU):
        super().__init__()
        self.input_size = input_size

        layers = []
        layers+=[nn.BatchNorm1d(input_size), nn.Linear(input_size, hidden_units), act_layer()]
        for i in range(hidden_layers-1):
            layers+=[nn.Linear(hidden_units, hidden_units), act_layer()]
        layers+=[nn.Linear(hidden_units, output_size), nn.Tanh()]

        self.layers = nn.ModuleList(layers)

        print("init policy network...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, state1, state2):
        state1 = list(state1)
        state2 = list(state2)
        B, _, _ = state1[0].shape
        
        for i in range(len(state1)):
            if i == 0:
                merge1 = state1[0].reshape(B, -1)
                merge2 = state2[0].reshape(B, -1)
            else:
                merge1 =torch.cat((merge1, state1[i].reshape(B, -1)), dim = -1)
                merge2 =torch.cat((merge2, state2[i].reshape(B, -1)), dim = -1)
        x = torch.cat((merge1, merge2), dim = -1)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(B, -1, 3)
        return x

class World_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = 5, hidden_units = 1024, act_layer = nn.ELU):
        super().__init__()
        self.input_size = input_size

        layers = []
        layers+=[nn.BatchNorm1d(input_size), nn.Linear(input_size, hidden_units), act_layer()]
        for i in range(hidden_layers-1):
            layers+=[nn.Linear(hidden_units, hidden_units), act_layer()]
        layers+=[nn.Linear(hidden_units, output_size)]

        self.layers = nn.ModuleList(layers)

        print("init world network...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, state, T_rot, T_ang):
        state = list(state)
        B, _, _ = state[0].shape
        for i in range(len(state)):
            if i == 0:
                merge1 = state[0].reshape(B, -1)
            else:
                merge1 =torch.cat((merge1, state[i].reshape(B, -1)), dim = -1)
        x = torch.cat((merge1, T_rot.reshape(B, -1), T_ang.reshape(B, -1)), dim = -1)
        for layer in self.layers:
            x = layer(x)

        pos_a = x[:, x.shape[-1]//2:].reshape(B, -1, 3)
        rot_a = x[:, :x.shape[-1]//2].reshape(B, -1, 3)
        return pos_a, rot_a


class SuperTrack:
    def __init__(self, device):
        self.world = World_Model(502, 132).to(device)
        self.world_batchsize = 2048
        self.world_lr = 0.001
        self.world_window = 1
        self.world_optimizer = optim.RAdam(self.world.parameters(), lr=self.world_lr)

        self.policy = Policy_Model(710, 63).to(device)
        self.policy_batchsize = 1024
        self.policy_lr = 0.0001
        self.policy_window = 2
        self.policy_optimizer = optim.RAdam(self.policy.parameters(), lr=self.policy_lr)

        self.EPOCH = 200
        self.iterations = 10000
        self.device = device

        print("preparing data...")
        self.dataloader = BVHdataset(window_size=max(self.policy_window, self.world_window)).get_loader(max(self.world_batchsize, self.policy_batchsize), num_worker=1, shuffle=True)
        print("done")
        # print("preparing data...")
        # self.policy_dataloader = BVHdataset(window_size=self.policy_window).get_loader(self.policy_batchsize, num_worker=1)
        # print("done")


        self.dtime = 1/30

    
    @lru_cache
    def local(self, pos, vel, rot, ang):
        '''
        convert world space input into local space

        pos:[B, body, 3] position in world space
        vel:[B, body, 3] velocity in world space
        rot:[B, body, 4] rotation in world space, quaternion
        ang:[B, body, 3] rotational velocity in world space, axis-angle
        '''
        
        BS, B, _ = pos.shape
        
        rootr = pyt.quaternion_to_matrix(rot[:, 0:1, :])
        inv = torch.transpose(rootr, -1, -2)# [B, 1, 3, 3]
        lpos = torch.matmul(inv, pos.unsqueeze(-1)).squeeze(-1)
        lvel = torch.matmul(inv, vel.unsqueeze(-1)).squeeze(-1)
        lrot = pyt.matrix_to_rotation_6d(torch.matmul(inv, pyt.quaternion_to_matrix(rot)))
        lang = torch.matmul(inv, ang.unsqueeze(-1)).squeeze(-1)
        height = pos[:, :, 1]
        lup = torch.matmul(inv, torch.tensor([[0], [0], [1]], dtype = inv.dtype).unsqueeze(0).unsqueeze(0).repeat(BS, 1, 1, 1).to(self.device))
        lup = lup.squeeze(1).squeeze(-1)
        
        # print(pos.shape, vel.shape, rot.shape, ang.shape)
        # print(lpos.shape, lvel.shape, lrot.shape, lang.shape, height.shape, lup.shape)
        return lpos, lvel, lrot, lang, height, lup
    
    

    def forward(self, pos, vel, rot, ang, j_rot, j_ang, type = 'world'):
        '''
        pos: [B, frames+1, body, 3]
        vel: [B, frames+1, body, 3]
        rot: [B, frames+1, body, 4] quaternion
        ang: [B, frames+1, body, 3] axis-angle
        j_rot: [B, frames+1 joints, 4] quaternion
        j_ang: [B, frames+1, joints, 3] axis_angle
        type: 'world'|'policy'
        '''

        #initial state, S0<-P0
        s_pos = pos[:, 0, :, :] # [B, body, 3]
        s_vel = vel[:, 0, :, :] # [B, body, 3]
        s_rot = rot[:, 0, :, :] # [B, body, 4]
        s_ang = ang[:, 0, :, :] # [B, body, 3]

        loss = 0

        for i in range(self.world_window):
            #produce o:[B, joint, 3] represented as axis-angle
            o = self.policy(self.local(s_pos, s_vel, s_rot, s_ang), self.local(pos[:, i+1, :, :], vel[:, i+1, :, :], rot[:, i+1, :, :], ang[:, i+1, :, :]))
            o_hat = o+0.1*torch.randn_like(o) #add noise
            o_hat = 120*o_hat # times alpha

            # get PD target
            T_rot = j_rot[:, i+1, :, :] #[B, joints, 4]
            T_rot = pyt.quaternion_to_matrix(T_rot)
            o_hat = pyt.axis_angle_to_matrix(o_hat)
            T_rot = torch.matmul(o_hat, T_rot)
            T_rot = pyt.matrix_to_quaternion(T_rot)
            T_ang = j_ang[:, i+1, :, :]
            
            # get accelerations, both shapes are [B, body, 3]
            position_acceleration, rotation_acceleration = self.world(self.local(s_pos, s_vel, s_rot, s_ang), T_rot, T_ang)

            # turn accelerations into world space 
            rootr = pyt.quaternion_to_matrix(s_rot[:, 0:1, :]) # [B, body, 3, 3]
            position_acceleration = torch.matmul(rootr, position_acceleration.unsqueeze(-1)).squeeze(-1)
            rotation_acceleration = torch.matmul(rootr, rotation_acceleration.unsqueeze(-1)).squeeze(-1)
            
            #update
            
            # s_vel = vel[:, i, :, :]
            # s_ang = ang[:, i, :, :]
            # because of data, change the order
            s_pos = s_pos + self.dtime*s_vel
            s_rot = pyt.matrix_to_quaternion(torch.matmul(pyt.axis_angle_to_matrix(s_ang*self.dtime), pyt.quaternion_to_matrix(s_rot)))
            s_vel = s_vel + self.dtime*position_acceleration
            s_ang = s_ang + self.dtime*rotation_acceleration

            if type == 'world':
                loss+=self.train_world_loss(pos[:, i+1, :, :], s_pos, vel[:, i+1, :, :], s_vel, rot[:, i+1, :, :], s_rot, ang[:, i+1, :, :], s_ang)
            
            if type == 'policy':
                lpos1, lvel1, lrot1, lang1, height1, lup1 = self.local(s_pos, s_vel, s_rot, s_ang)
                lpos2, lvel2, lrot2, lang2, height2, lup2 = self.local(pos[:, i+1, :, :], vel[:, i+1, :, :], rot[:, i+1, :, :], ang[:, i+1, :, :])
                loss+=self.train_policy_loss(lpos1, lpos2, lvel1, lvel2, lrot1, lrot2, lang1, lang2, height1, height2, lup1, lup2, o)

        # pdb.set_trace()
        return loss

            
    def train_world_loss(self, pos1, pos2, vel1, vel2, rot1, rot2, ang1, ang2):
        wpos = wvel = wrot = wang = 1
        loss = wpos*torch.mean(torch.sum(torch.abs(pos1-pos2), dim = -1))
        loss += wvel*torch.mean(torch.sum(torch.abs(vel1-vel2), dim = -1))
        loss += wang*torch.mean(torch.sum(torch.abs(ang1-ang2), dim = -1))
        loss += wrot*torch.mean(torch.abs(torch.arccos(pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2))[:, :, 0]*0.99)))

        # print(wpos*torch.mean(torch.sum(torch.abs(pos1-pos2), dim = -1)))
        # print(wvel*torch.mean(torch.sum(torch.abs(vel1-vel2), dim = -1)))
        # print(wang*torch.mean(torch.sum(torch.abs(ang1-ang2), dim = -1)))
        # print(wrot*torch.mean(torch.arccos(pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2))[:, :, 0]*0.99)))
        # print(torch.arccos(pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2))[:, :, 0]*0.99))

        return loss

    def train_policy_loss(self, lpos1, lpos2, lvel1, lvel2, lrot1, lrot2, lang1, lang2, hei1, hei2, up1, up2, o):
        wlpos = wlvel = wlrot = wlang = whei = wup = 0.1
        wlreg = wsreg = 0.01
        
        loss = wlpos*torch.mean(torch.sum(torch.abs(lpos1-lpos2), dim = -1))
        loss += wlvel*torch.mean(torch.sum(torch.abs(lvel1-lvel2), dim = -1))
        loss += wlrot*torch.mean(torch.sum(torch.abs(lrot1-lrot2), dim = -1))
        loss += wlang*torch.mean(torch.sum(torch.abs(lang1-lang2), dim = -1))
        loss += whei*torch.mean(torch.sum(torch.abs(hei1-hei2), dim = -1))
        loss += wup*torch.mean(torch.sum(torch.abs(up1-up2), dim = -1))
        loss += wlreg*torch.mean(torch.sum(torch.abs(o), dim = -1))
        loss += wsreg*torch.mean(torch.sum(torch.square(o), dim = -1))

        # print(wlpos*torch.mean(torch.sum(torch.abs(lpos1-lpos2), dim = -1)))
        # print(wlvel*torch.mean(torch.sum(torch.abs(lvel1-lvel2), dim = -1)))
        # print(wlrot*torch.mean(torch.sum(torch.abs(lrot1-lrot2), dim = -1)))
        # print(wlang*torch.mean(torch.sum(torch.abs(lang1-lang2), dim = -1)))
        # print(whei*torch.mean(torch.abs(hei1-hei2)))
        # print(wup*torch.mean(torch.sum(torch.abs(up1-up2), dim = -1)))
        # print(wlreg*torch.mean(torch.sum(torch.abs(o), dim = -1)))
        # print(wsreg*torch.mean(torch.sum(torch.square(o), dim = -1)))

        return loss



    def train(self):
        writer = SummaryWriter('./tensorboard')#使用tensorboard
        try:
            os.makedirs('./model')
        except OSError:
            pass

        for epoch in range(self.EPOCH):
            world_data_index = [i for i in range(self.world_window+1)]
            policy_data_index = [i for i in range(self.policy_window+1)]
            world_batch_index = random.sample(range(max(self.world_batchsize, self.policy_batchsize)), self.world_batchsize)
            policy_batch_index = random.sample(range(max(self.world_batchsize, self.policy_batchsize)), self.policy_batchsize)
            for data in self.dataloader:
                pos1 = data[0][world_batch_index, :, :, :][:, world_data_index, :, :].to(self.device).to(torch.float32)
                vel1 = data[1][world_batch_index, :, :, :][:, world_data_index, :, :].to(self.device).to(torch.float32)
                rot1 = data[2][world_batch_index, :, :, :][:, world_data_index, :, :].to(self.device).to(torch.float32)
                ang1 = data[3][world_batch_index, :, :, :][:, world_data_index, :, :].to(self.device).to(torch.float32)
                j_rot1 = data[4][world_batch_index, :, :, :][:, world_data_index, :, :].to(self.device).to(torch.float32)
                j_ang1 = data[5][world_batch_index, :, :, :][:, world_data_index, :, :].to(self.device).to(torch.float32)

                pos2 = data[0][policy_batch_index, :, :, :][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                vel2 = data[1][policy_batch_index, :, :, :][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                rot2 = data[2][policy_batch_index, :, :, :][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                ang2 = data[3][policy_batch_index, :, :, :][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                j_rot2 = data[4][policy_batch_index, :, :, :][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                j_ang2 = data[5][policy_batch_index, :, :, :][:, policy_data_index, :, :].to(self.device).to(torch.float32)

                self.world.train()
                self.policy.train()

                self.world.train()
                for param in self.world.parameters():
                    param.requires_grad =True
                self.policy.eval()
                for param in self.policy.parameters():
                    param.requires_grad =False
                loss = self.forward(pos1, vel1, rot1, ang1, j_rot1, j_ang1, type='world')
                print('world loss: ', loss)
                self.world_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.world_optimizer.step()
                del loss

                self.world.eval()
                for param in self.world.parameters():
                    param.requires_grad =False
                self.policy.train()
                for param in self.policy.parameters():
                    param.requires_grad =True
                loss = self.forward(pos2, vel2, rot2, ang2, j_rot2, j_ang2, type='policy')
                print('policy loss: ', loss)
                self.world_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                del loss

if __name__  == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = SuperTrack(device)
    a.local(torch.rand(33, 22, 3).to(device), torch.rand(33, 22, 3).to(device), torch.rand(33, 22, 4).to(device), torch.rand(33, 22, 3).to(device))
    b = a.forward(torch.rand(5, 33, 22, 3), torch.rand(5, 33, 22, 3), torch.rand(5, 33, 22, 4), torch.rand(5, 33, 22, 3), torch.rand(5, 33, 21, 4), torch.rand(5, 33, 21, 3))