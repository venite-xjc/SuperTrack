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
from tqdm import tqdm
import numpy as np

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        # pdb.set_trace()
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]



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
        self.world_window = 8
        self.world_optimizer = optim.RAdam(self.world.parameters(), lr=self.world_lr)
        
        self.policy = Policy_Model(710, 63).to(device)
        self.policy_batchsize = 1024
        self.policy_lr = 0.0001
        self.policy_window = 32
        self.policy_optimizer = optim.RAdam(self.policy.parameters(), lr=self.policy_lr)
        
        self.EPOCH = 200
        self.iterations = 10000
        self.device = device

        world_state_dict = torch.load('/root/SuperTrack/model1/epoch_119_world.pth')
        policy_state_dict = torch.load('/root/SuperTrack/model1/epoch_119_policy.pth')
        self.world.load_state_dict(world_state_dict)
        self.policy.load_state_dict(policy_state_dict)

        self.dtime = 1/30

    def TwoAxis(self, matrix): # y-axis is upforward
        batch_dim = matrix.size()[:-2]
        return matrix[..., :, [0, 2]].clone().reshape(batch_dim + (6,))

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
        lrot = self.TwoAxis(torch.matmul(inv, pyt.quaternion_to_matrix(rot)))
        lang = torch.matmul(inv, ang.unsqueeze(-1)).squeeze(-1)
        height = pos[:, :, 1]
        lup = torch.matmul(inv, torch.tensor([[0], [1], [0]], dtype = inv.dtype).unsqueeze(0).unsqueeze(0).repeat(BS, 1, 1, 1).to(self.device))
        lup = lup.squeeze(1).squeeze(-1)
        
        # print(pos.shape, vel.shape, rot.shape, ang.shape)
        # print(lpos.shape, lvel.shape, lrot.shape, lang.shape, height.shape, lup.shape)
        return lpos, lvel, lrot, lang, height, lup
    
    

    def forward(self, pos, vel, rot, ang, j_rot, j_ang, type = 'world', return_ = 'loss'):
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
        output_pos = []
        if type == 'world':
            window = self.world_window
        if type == 'policy':
            window = self.policy_window
        for i in range(window):
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
            
            # because of data, change the order
            s_pos = s_pos + self.dtime*s_vel #[B, body, 3]
            s_rot = pyt.matrix_to_quaternion(torch.matmul(pyt.axis_angle_to_matrix(s_ang*self.dtime), pyt.quaternion_to_matrix(s_rot)))
            s_vel = s_vel + self.dtime*position_acceleration
            s_ang = s_ang + self.dtime*rotation_acceleration

            if return_ == "loss":
                if type == 'world':
                    loss+=self.train_world_loss(pos[:, i+1, :, :], s_pos, vel[:, i+1, :, :], s_vel, rot[:, i+1, :, :], s_rot, ang[:, i+1, :, :], s_ang)
                
                if type == 'policy':
                    lpos1, lvel1, lrot1, lang1, height1, lup1 = self.local(s_pos, s_vel, s_rot, s_ang)
                    lpos2, lvel2, lrot2, lang2, height2, lup2 = self.local(pos[:, i+1, :, :], vel[:, i+1, :, :], rot[:, i+1, :, :], ang[:, i+1, :, :])
                    loss+=self.train_policy_loss(lpos1, lpos2, lvel1, lvel2, lrot1, lrot2, lang1, lang2, height1, height2, lup1, lup2, o)
            if return_ == 'pos':
                output_pos.append(s_pos)
        # pdb.set_trace()
        if return_ == "loss":
            return loss
        if return_ == "pos":
            return torch.cat(output_pos)

            
    def train_world_loss(self, pos1, pos2, vel1, vel2, rot1, rot2, ang1, ang2):
        wpos = wvel = wrot = wang = 0.1
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
        wlreg = wsreg = 0.02
        
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

        print("preparing data...")
        self.train_dataloader = BVHdataset(window_size=max(self.policy_window, self.world_window)).get_loader(max(self.world_batchsize, self.policy_batchsize), num_worker=4, shuffle=True)
        print("done")
        print("preparing data...")
        self.test_dataloader = BVHdataset(window_size=max(self.policy_window, self.world_window), type_ ="test").get_loader(500, num_worker=4, shuffle=True)
        print("done")

        self.world_scheduler = WarmupPolyLR(self.world_optimizer, max_iters=len(self.train_dataloader)*self.EPOCH, power=0.9, warmup_iters=len(self.train_dataloader)*2)
        self.policy_scheduler = WarmupPolyLR(self.policy_optimizer, max_iters=len(self.train_dataloader)*self.EPOCH, power=0.9, warmup_iters=len(self.train_dataloader)*2)


        for epoch in range(self.EPOCH):
            world_data_index = [i for i in range(self.world_window+1)]
            policy_data_index = [i for i in range(self.policy_window+1)]
            world_batch_index = random.sample(range(max(self.world_batchsize, self.policy_batchsize)), self.world_batchsize)
            policy_batch_index = random.sample(range(max(self.world_batchsize, self.policy_batchsize)), self.policy_batchsize)
            
            world_lr = self.world_optimizer.param_groups[0]['lr']
            policy_lr = self.policy_optimizer.param_groups[0]['lr']

            world_loss = []
            policy_loss = []
            for data in tqdm(self.train_dataloader):
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
                # print('world loss: ', loss)
                self.world_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.world.parameters(), 25.0)
                self.world_optimizer.step()
                self.world_scheduler.step()
                world_loss.append(loss.detach().cpu().numpy().item())
                del loss
                

                self.world.eval()
                for param in self.world.parameters():
                    param.requires_grad =False
                self.policy.train()
                for param in self.policy.parameters():
                    param.requires_grad =True
                loss = self.forward(pos2, vel2, rot2, ang2, j_rot2, j_ang2, type='policy')
                # print('policy loss: ', loss)
                self.world_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 100.0)
                self.policy_optimizer.step()
                self.policy_scheduler.step()
                policy_loss.append(loss.detach().cpu().numpy().item())
                del loss

            world_test_loss = []
            policy_test_loss = []
            for data in tqdm(self.test_dataloader):
                pos1 = data[0][:, world_data_index, :, :].to(self.device).to(torch.float32)
                vel1 = data[1][:, world_data_index, :, :].to(self.device).to(torch.float32)
                rot1 = data[2][:, world_data_index, :, :].to(self.device).to(torch.float32)
                ang1 = data[3][:, world_data_index, :, :].to(self.device).to(torch.float32)
                j_rot1 = data[4][:, world_data_index, :, :].to(self.device).to(torch.float32)
                j_ang1 = data[5][:, world_data_index, :, :].to(self.device).to(torch.float32)

                pos2 = data[0][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                vel2 = data[1][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                rot2 = data[2][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                ang2 = data[3][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                j_rot2 = data[4][:, policy_data_index, :, :].to(self.device).to(torch.float32)
                j_ang2 = data[5][:, policy_data_index, :, :].to(self.device).to(torch.float32)

                self.world.eval()
                self.policy.eval()

                loss = self.forward(pos1, vel1, rot1, ang1, j_rot1, j_ang1, type='world')
                world_test_loss.append(loss.detach().cpu().numpy().item())
                del loss
                
                loss = self.forward(pos2, vel2, rot2, ang2, j_rot2, j_ang2, type='policy')
                policy_test_loss.append(loss.detach().cpu().numpy().item())
                del loss
            
            print("-----|EPOCH %3d|world loss: %13.8f|policy loss: %13.8f|world test loss: %13.8f|policy test loss: %13.8f|------"%(int(epoch), np.mean(np.array(world_loss)), np.mean(np.array(policy_loss)), np.mean(np.array(world_test_loss)), np.mean(np.array(policy_test_loss))))
            writer.add_scalar('train_world_loss', np.mean(np.array(world_loss)), epoch)
            writer.add_scalar('train_policy_loss', np.mean(np.array(policy_loss)), epoch)
            writer.add_scalar('test_world_loss', np.mean(np.array(world_test_loss)), epoch)
            writer.add_scalar('test_policy_loss', np.mean(np.array(policy_test_loss)), epoch)
            writer.add_scalar('train_lr', world_lr, epoch)
            writer.add_scalar('policy_lr', policy_lr, epoch)
            if epoch%10==9:
                print("save models")
                torch.save(self.world.state_dict(), './model/epoch_{}_world.pth'.format(epoch))
                torch.save(self.policy.state_dict(), './model/epoch_{}_policy.pth'.format(epoch))
    
if __name__  == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = SuperTrack(device)
    a.local(torch.rand(33, 22, 3).to(device), torch.rand(33, 22, 3).to(device), torch.rand(33, 22, 4).to(device), torch.rand(33, 22, 3).to(device))
    b = a.forward(torch.rand(5, 33, 22, 3), torch.rand(5, 33, 22, 3), torch.rand(5, 33, 22, 4), torch.rand(5, 33, 22, 3), torch.rand(5, 33, 21, 4), torch.rand(5, 33, 21, 3))