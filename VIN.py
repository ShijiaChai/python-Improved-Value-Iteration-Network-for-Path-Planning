import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def attention(tensor, params):
    """Attention model for grid world
    """
    S1, S2, args = params

    num_data = tensor.size()[0]

    # Slicing S1 positions
    slice_s1 = S1.expand(args.imsize, 1, args.ch_q, num_data)
    slice_s1 = slice_s1.permute(3, 2, 1, 0)
    q_out = tensor.gather(2, slice_s1).squeeze(2)

    # Slicing S2 positions
    slice_s2 = S2.expand(1, args.ch_q, num_data)
    slice_s2 = slice_s2.permute(2, 1, 0)
    q_out = q_out.gather(2, slice_s2).squeeze(2)

    return q_out

class duel_layer(nn.Module):
    def __init__(self, args):
        super (duel_layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=args.ch_q, out_features=1, bias=False),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=args.ch_q, out_features=8, bias=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        a = torch.zeros(1)
        b = torch.zeros(8)
        a = Variable(a)
        b = Variable(b)
        a = a.cuda()
        b = b.cuda()
        a = self.layer1(x)
        b = self.layer2(x)
        output = a + (b - torch.mean(b, dim=1).reshape(-1, 1))
        return output

class VImodel(nn.Module):
    """Value Iteration Model"""
    def __init__(self):
        super(VImodel, self).__init__()

        # First hidden Conv layer
        self.conv_h = nn.Conv2d(in_channels=2,
                                out_channels=150,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1) // 2,  # SAME padding: (F - 1)/2
                                bias=False)
        # Conv layer to generate reward image
        self.conv_r = nn.Conv2d(in_channels=150,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1) // 2,  # SAME padding: (F - 1)/2
                                bias=False)
        # q layers in VI module
        self.conv_q = nn.ModuleList([nn.Conv2d(in_channels=2,  # stack [r, v] -> 2 channels
                                out_channels=10,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1) // 2,  # SAME padding: (F - 1)/2
                                bias=False) for i in range(16)])

        # Record grid image, reward image and its value images for each VI iteration

    def forward(self, X):
        # Get reward image from observation image
        h = self.conv_h(X)
        r = self.conv_r(h)
        # Initialize value map (zero everywhere)
        v = torch.zeros(r.size())
        # Move to GPU if necessary
        v = v.cuda() if X.is_cuda else v
        # Wrap to autograd.Variable
        v = Variable(v)
        VIN_sum = torch.zeros(r.size())
        VIN_sum = VIN_sum.cuda() if X.is_cuda else VIN_sum
        VIN_sum = Variable(VIN_sum)
        # K-iterations of Value Iteration module
        for i, _ in enumerate(self.conv_q):
            rv = torch.cat([r, v], 1)  # [batch_size, 2, imsize, imsize]
            q = self.conv_q[i](rv)
            v, _ = torch.max(q, 1, keepdim=True)  # torch.max returns (values, indices)
            VIN_sum += v
        # output the v-value:
        v = torch.div(VIN_sum, 4)
        return v


class HVIN(nn.Module):
    """Value Iteration Network architecture"""

    def __init__(self, args):
        super(HVIN, self).__init__()

        # First hidden Conv layer
        self.conv_h = nn.Conv2d(in_channels=args.ch_i,
                                out_channels=args.ch_h,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1) // 2,  # SAME padding: (F - 1)/2
                                bias=False)
        # Conv layer to generate reward image
        self.conv_r = nn.Conv2d(in_channels=args.ch_h,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1) // 2,  # SAME padding: (F - 1)/2
                                bias=False)
        # q layers in VI module
        self.conv_q = nn.ModuleList([nn.Conv2d(in_channels=3,  # stack [new_r, v] -> 3 channels
                                out_channels=args.ch_q,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1) // 2,  # SAME padding: (F - 1)/2
                                bias=False) for i in range(args.k)])
        # BN layers
        self.BNlayer = nn.BatchNorm2d(args.ch_q)
        # Final fully connected layer
        self.duel = duel_layer(args)
        # VI model
        self.VI = VImodel()
        # Max pooling
        self.pooling = nn.MaxPool2d(2, 2)
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # Record grid image, reward image and its value images for each VI iteration
        self.grid_image = None
        self.reward_image = None
        self.value_images = []

    def forward(self, X, S1, S2, args, record_images=False):
        # Down sampling
        X1 = self.pooling(X)
        # Get reward image from observation image
        h = self.conv_h(X)
        r = self.conv_r(h)
        # Computing HVIN
        v1 = self.VI(X1)
        # Up-sampling
        m = self.upsample(v1)
        new_r = torch.cat([m, r], 1)
        if record_images:  # TODO: Currently only support single input image
            # Save grid image in Numpy array
            self.grid_image = X.data[0].cpu().numpy()  # cpu() works both GPU/CPU mode
            # Save reward image in Numpy array
            self.reward_image = r.data[0].cpu().numpy()  # cpu() works both GPU/CPU mode
        # Initialize value map (zero everywhere)
        v = torch.zeros(new_r.size(0), 1, new_r.size(2), new_r.size(3))
        # Move to GPU if necessary
        v = v.cuda() if X.is_cuda else v
        # Wrap to autograd.Variable
        v = Variable(v)
        VIN_sum = torch.zeros(new_r.size(0), 1, new_r.size(2), new_r.size(3))
        VIN_sum = VIN_sum.cuda() if X.is_cuda else VIN_sum
        VIN_sum = Variable(VIN_sum)
        # K-iterations of Value Iteration module
        for i, _ in enumerate(self.conv_q):
            rv = torch.cat([new_r, v], 1)  # [batch_size, 2, imsize, imsize]
            q = self.conv_q[i](rv)
            v, _ = torch.max(q, 1, keepdim=True)  # torch.max returns (values, indices)
            VIN_sum += v
            if record_images:
                # Save single value image in Numpy array for each VI step
                self.value_images.append(v.data[0].cpu().numpy())  # cpu() works both GPU/CPU mode
        # Do one last convolution
        v = torch.div(VIN_sum, args.k)
        rv = torch.cat([new_r, v], 1)  # [batch_size, 3, imsize, imsize]
        q = self.conv_q[args.k-1](rv)
        q = self.BNlayer(q)
        # Attention model
        q_out = attention(q, [S1.long(), S2.long(), args])
        # Final Fully Connected layer
        logits = self.duel(q_out)
        return logits

