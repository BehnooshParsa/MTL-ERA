import torch
import torch.nn as nn

from util.Uwdatareader_UW import *


class GCNModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        A = torch.tensor(getAs(), dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.s_gcn_networks = nn.ModuleList((
            s_gcn(in_channels, 64, spatial_kernel_size, 1, residual=False),  # (N, 3, T, 15) -> (N,  64, T, 15)
            s_gcn(64, 64, spatial_kernel_size, 1),  # (N, 64, T, 15) -> (N,  64, T, 15)
            s_gcn(64, 64, spatial_kernel_size, 1),  # (N, 64, T, 15) -> (N,  64, T, 15)
            s_gcn(64, 64, spatial_kernel_size, 1),  # (N, 64, T, 15) -> (N,  64, T, 15)
            s_gcn(64, 128, spatial_kernel_size, 2),  # (N, 64, T, 15) -> (N,  128, T, 15)
            s_gcn(128, 128, spatial_kernel_size, 1),  # (N, 128, T, 15) -> (N,  128, T, 15)
            s_gcn(128, 128, spatial_kernel_size, 1),  # (N, 128, T, 15) -> (N,  128, T, 15)
            s_gcn(128, 256, spatial_kernel_size, 2),  # (N, 128, T, 15) -> (N,  256, T, 15)
            s_gcn(256, 256, spatial_kernel_size, 1),  # (N, 256, T, 15) -> (N,  256, T, 15)
            s_gcn(256, 256, spatial_kernel_size, 1),  # (N, 256, T, 15) -> (N,  256, T, 15)
        ))

        self.adaptpool = nn.AdaptiveAvgPool2d((15, 2))

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for gcn in self.s_gcn_networks:
            x, _ = gcn(x, self.A)
        # y = F.adaptive_avg_pool2d(x.permute(0,2,1,3), x.size()[3:])
        # x = x.view(N, M, -1, 1, 1).mean(dim=1)
        y = x.permute(0, 2, 1, 3).contiguous() # (N, 256, T, 15) -> (N,  T, 256, 15)
        z = y.view(N, T, -1) # (N,  T, 256, 15) -> (N,  T, 3840)
        # y=self.adaptpool(x.permute(0,2,1,3))
        return z


class s_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1), padding=(0, 0),
                              stride=(1, 1), dilation=(1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return self.relu(x.contiguous()), A
