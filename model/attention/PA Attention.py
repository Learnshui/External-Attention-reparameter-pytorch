# -*- coding: utf-8 -*-
import torch
import math
from torch.nn import Module, Conv2d, Parameter,Sigmoid, Softmax
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'PA_Module']
#attention

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width) #view函数相当于resize的功能，将原来的tensor变换成指的维度

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PA_Module(Module):
    """ Phase attention module"""
    def __init__(self, in_dim):
        super(PA_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()
        self.softmax11 = Softmax()
    def forward(self,x_fix,x_ref):
        """ calculate dependency among x_fix and x_ref, adaptively select x_ref according to the correlation of x_ref and x_fix
            计算x_fix和x_ref之间的依赖关系，根据x_ref和x_fix的相关性自适应选择x_ref
            inputs :
                x_fix : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x_ref.size()
        proj_ref = x_ref.view(m_batchsize, C, -1)# (m_batchsize, C,height*width）
        proj_fix = x_fix.view(m_batchsize, C, -1).permute(0, 2, 1) #(m_batchsize,height*width, C）
        #计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),
        # 注意两个tensor的维度必须为3，结果（b,h,h）.
        energy = torch.bmm(proj_ref, proj_fix)
        #将输入tensor的维度扩展为与指定tensor相同的size
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy#这是啥意思
        attention = self.softmax(energy_new)
        channel_scale = torch.sum(attention, 2)#对第三个维度进行求和，少了一个维度
        channel_scale = channel_scale.unsqueeze(-1)#增加列维度
        channel_scale = self.sigmoid(channel_scale)

        proj_value = x_ref.view(m_batchsize, C, -1)  # reshape
        out = channel_scale * proj_value
        out = out.view(m_batchsize, C, height, width)  # reshape
        out = self.gamma * out + x_ref
        #与图文不符啊感觉

        return out
model = PA_Module(3) #**对应的变量是可变长度的字典变量
# print(model)
