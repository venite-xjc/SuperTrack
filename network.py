import torch
import torch.nn as nn
import torch.optim as optim

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



    def train_world(self, state_list):
        '''
        state_list: [B, W+1, N]
        '''
        assert state_list.shape[1] == self.world_window+1

        state_tmp = state_list[:, 0, :]
        state_pred = []
        for i in range(self.world_window):
            a = self.world(state_tmp)
            state_tmp = self.update_state(state_tmp, a)
            state_pred.append(state_tmp)
            
