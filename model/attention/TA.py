import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)
class ChannelMeanAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelMeanAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias = True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias = True)
        self.relu = nonlinearity
        
    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = F.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class ChannelMeanMaxAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelMeanMaxAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias = True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias = True)
        self.relu = nonlinearity

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor_mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1_mean = self.relu(self.fc1(squeeze_tensor_mean))
        fc_out_2_mean = self.fc2(fc_out_1_mean)
        
        squeeze_tensor_max = input_tensor.view(batch_size, num_channels, -1).max(dim=2)[0]
        fc_out_1_max = self.relu(self.fc1(squeeze_tensor_max))
        fc_out_2_max = self.fc2(fc_out_1_max) 
 
        a, b = squeeze_tensor_mean.size()
        result = torch.Tensor(a,b)
        result = torch.add(fc_out_2_mean, fc_out_2_max)
        fc_out_2 = F.sigmoid(result)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding =3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x
        avg_out = torch.mean(x, dim=1, keepdim = True)
        max_out, _ = torch.max(x, dim=1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim =1)
        x = self.conv1(x)
        return self.sigmoid(x) * input_tensor
class TA_(nn.Module):
    def __init__(self, num_classes=2, num_channels=3):
        super(TA_, self).__init__()
        filters = [64, 128, 256, 512]
        self.channelmeanmaxattention1 = ChannelMeanMaxAttention(filters[2]*2)
        self.spatialattention1 = SpatialAttention()
    def forward(self, x):
    
        d = self.channelmeanmaxattention1(x)
        d = self.spatialattention1(d)

        return F.sigmoid(d)

if __name__ == '__main__':
    input=torch.randn(8,512,49,49)
    block=TA_()
    out=block(input)
    print(out.shape)
