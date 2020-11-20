import torch
from torch import nn
import torch.nn.functional as F
from config import get_config, print_usage
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
'''
main network-----PESA_Net
'''

def batch_episym(x1, x2, F):  # This code is very important
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
    return ys

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, weight):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(weight))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class SSEU(Module):
    def __init__(self, in_channels, channels,  bias=True, radix=2, reduction_factor=4, **kwargs):
        super(SSEU, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor, 32)  # begin:128,here is 32
        self.radix = radix
        self.channels = channels
        self.conv = Conv2d(in_channels, channels*radix, kernel_size=1, stride=1,  # 1*1？
                               groups=radix, bias=bias, **kwargs)
        self.in0 = nn.InstanceNorm2d(channels*radix, eps=1e-5)
        self.bn0 = nn.BatchNorm2d(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.in0(x)    # add IN
        x = self.bn0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            out1 ={}
            for ii in range(len(splited)):
                gap = splited[ii]
                gap = F.adaptive_avg_pool2d(gap, 1)  # adaptive_avg_pool
                gap = self.fc1(gap)  # 64
                gap = self.bn1(gap)
                gap = self.relu(gap)  # activation function
                atten = self.fc2(gap)  # finally we conv again
                atten = torch.sigmoid(atten).view(batch, -1, 1, 1)  # activation function
                out1[ii] = atten * gap
            out = torch.cat([out1[_I] for _I in range(len(out1))],dim=1)
        else:
            gap = x
            gap = F.adaptive_avg_pool2d(gap, 1) # adaptive_avg_pool
            gap = self.fc1(gap)  # 64
            gap = self.bn1(gap)
            gap = self.relu(gap)  # activation function
            atten = self.fc2(gap)   # finally we conv again
            atten = torch.sigmoid(atten).view(batch, -1, 1, 1) # activation function
            out = atten * x
        return out.contiguous()

class PESA_Block(nn.Module):     #config.split, config.group
    expansion = 2
    def __init__(self, inplanes, planes, stride=1,
                 radix=2,  bottleneck_width=64 ):
        super(PESA_Block, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.))
        print('radix:',radix)
        self.shot_cut = None
        if planes*2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes*2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1,groups=radix, bias=True)
        self.in1 = nn.InstanceNorm2d(group_width, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        if radix >= 1:
            self.conv2 = SSEU(
                group_width, group_width
               , bias=True,
                radix=radix
                )
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, bias=False)
            self.in2 = nn.InstanceNorm2d(group_width, eps=1e-5)
            self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, planes*2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(planes*2, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)   # inplace=True don't impact the result, what more this operation can save memory

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.bn1(out) 
        out = self.relu(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.in2(out)
            out = self.bn2(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.in3(out)
        out = self.bn3(out)
        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out

class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_geo, x_down):
        embed = self.conv(x_geo)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

class geo_attention_block(nn.Module):
    def __init__(self, channels, l2_nums,num):
        nn.Module.__init__(self)
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(num):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
    def forward(self, pre):
        x_down = self.down1(pre)      
        x2 = self.l2(x_down)
        x_geo = self.up1(pre, x2)        
        return x_geo

class PESA(nn.Module):
    def __init__(self, channels, input_channel, PESA_num, clusters, split, model_name):
        nn.Module.__init__(self)     
        self.PESA_num = PESA_num
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)
        self.in1 = nn.InstanceNorm2d(channels, eps=1e-5) # Context Norm has the same operation as Instance Norm.
        self.bn1 = nn.BatchNorm2d(channels)
        self.Re = nn.ReLU(inplace=True)
        self.pre = []
        for _ in range(self.PESA_num):
            self.pre.append(PESA_Block(channels,channels//2,1,split))
        self.geo = geo_attention_block(channels,clusters,3)
        self.post = []
        self.post.append(PESA_Block(2*channels, channels//2,1,split))
        for _ in range(self.PESA_num-1):
            self.post.append(PESA_Block(channels,channels//2,1,split))
        self.pre = nn.Sequential(*self.pre)
        self.post = nn.Sequential(*self.post)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs):

        batch_size, num_pts = data.shape[0], data.shape[2]
        pre = self.conv1(data)
        pre = self.Re(self.bn1(self.in1(pre)))        
        pre = self.pre(pre)
        x_geo = self.geo(pre)       
        out = self.post( torch.cat([pre,x_geo], dim=1))
        weight = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, weight)
        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)
        return weight, e_hat, residual

class PESA_Net():   # main network
    def __init__(self, config):
        self.Iteration = config.Iteration
        PESA_num = config.PESA_num
        self.SubNetwork1 = PESA(config.channels, 4, PESA_num, config.clusters,config.split,config.model)
        self.SubNetwork2 = PESA(config.channels, 6, PESA_num, config.clusters ,config.split,config.model)
    def forward(self, data):
        res_weight, res_e_hat = [], []
        weight, e_hat, residual = self.SubNetwork1(data['input'], data['xs']) # data['xs'] is used to compute residual.
        res_weight.append(weight), res_e_hat.append(e_hat)
        weight, e_hat, residual = self.SubNetwork2(torch.cat([data['input'], residual.detach(), torch.relu(torch.tanh(weight)).reshape(residual.shape).detach()], dim=1), data['xs'])
        res_weight.append(weight), res_e_hat.append(e_hat) 
        return res_weight, res_e_hat

        



