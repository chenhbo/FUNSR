import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from torch.nn import Conv1d,Conv2d


############Define ADL Discriminator network#######################
class Discriminator(nn.Module):

    def __init__(self,
    geometric_init=True,
    bias = 0.5,
    out_dim = 256):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(1, out_dim)

        self.fc2 = nn.Linear(out_dim, out_dim)

        self.fc3 = nn.Linear(out_dim, out_dim)

        self.fc4 = nn.Linear(out_dim, 1)


        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2= nn.Dropout(0.1)
        # self.dropout3= nn.Dropout(0.1)
        if geometric_init:
            torch.nn.init.constant_(self.fc1.bias, -bias)
            torch.nn.init.normal_(self.fc1.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            torch.nn.init.constant_(self.fc2.bias, -bias)
            torch.nn.init.normal_(self.fc2.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))


            torch.nn.init.constant_(self.fc3.bias, -bias)
            torch.nn.init.normal_(self.fc3.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))


            torch.nn.init.normal_(self.fc4.weight,mean=np.sqrt(np.pi) / np.sqrt(1), std=0.0001)
            torch.nn.init.constant_(self.fc4.bias, -bias)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x  

    def sdf(self, x):
        return self.forward(x) 
    
