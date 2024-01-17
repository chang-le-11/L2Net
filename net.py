# -*- coding: utf-8 -*-
# @Time : 2022/9/2 14:56
import torch
import torch.nn as nn
from kornia.filters import SpatialGradient

class convblock(nn.Module):
    def __init__(self, dim, dim1):
        super(convblock, self).__init__()
        self.norm = nn.BatchNorm2d(dim1)
        self.conv = nn.Conv2d(dim, dim1, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        # print(y.shape)
        return x * y

class block(nn.Module):
    def __init__(self, dim):
        super(block, self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class LKA(nn.Module):
    def __init__(self, dim):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        input = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = input * attn
        attn = input + attn

        return attn

class CB(nn.Module):
    def __init__(self, dim):
        super(CB, self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.dwconv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.block = block(dim)

    def forward(self, x):
        input = x
        x = self.act(self.norm(self.dwconv1(x)))
        x = self.act(self.norm(self.dwconv2(x)))
        x = self.block(x)
        out = input + x
        return out

class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()
        self.spatial = SpatialGradient('diff')
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :,  1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        return y

class edge_aware(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(edge_aware, self).__init__()
        self.conv = convblock(2, in_chans)
        self.ed = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.CA = CA(out_chans, reduction=16)

    def forward(self, x):
        x = self.conv(x)
        input = x
        x = self.ed(x)
        res = x
        x = self.CA(x)
        out = res * x
        out = input + out
        return out


class VAM(nn.Module):
    def __init__(self, dim):
        super(VAM, self).__init__()
        self.block = block(dim)
        self.CB = CB(dim)
        self.LKA = LKA(dim)

    def forward(self, x):
        input = x
        x = self.block(x)
        x = self.LKA(x)
        u = self.CB(x)
        return u



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = [32, 64, 128]
        self.conv1 = convblock(2, dim[0])
        self.conv2 = convblock(dim[0], dim[1])
        self.VAM1 = VAM(dim[1])
        self.VAM2 = VAM(dim[1])
        self.VAM3 = VAM(dim[1])
        self.VAM4 = VAM(dim[1])
        self.VAM5 = VAM(dim[1])
        self.VAM6 = VAM(dim[1])

        self.ed_aware = edge_aware(dim[1], dim[1])
        self.edge = Edge()
        self.conv3 = convblock(dim[2], dim[1])
        self.conv4 = nn.Conv2d(dim[1], 1, 3, 1, 1)

    def forward(self, vis, ir):
        input = torch.cat([vis[:, :1], ir], 1)

        x = self.conv1(input)
        x = self.conv2(x)

        f = self.VAM1(x)
        f = self.VAM2(f)
        f = self.VAM3(f)
        f = self.VAM4(f)
        f = self.VAM5(f)
        f = self.VAM6(f)

        vi_ed = self.edge(vis[:, :1])
        ir_ed = self.edge(ir)
        edge = torch.cat([vi_ed, ir_ed], 1)
        edge = self.ed_aware(edge)
        out = torch.cat([edge, f], 1)
        # out = f
        out = self.conv3(out)
        out = torch.tanh(self.conv4(out)) / 2 + 0.5

        return out





if __name__ == '__main__':
    net = Net()
    a = torch.randn((1, 1, 256, 256))
    b = torch.randn((1, 1, 256, 256))
    # net1 = Edge()
    out = net(a, b)
    # print(net)
    # out = net1(a)
    print(out.shape)

