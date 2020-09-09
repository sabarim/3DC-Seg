from torch import nn
from torch.nn import functional as F

import math
import torch


class NonLocalBlock3DWithDownsampling(nn.Module):
    def __init__(self, in_channels, intermediate_channels, downsampling_factor, out_channels=None):
        super(self.__class__, self).__init__()

        self.theta = nn.Conv3d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)

        self.g = nn.Conv3d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)

        if not out_channels:
            out_channels = in_channels
        self.W = nn.Conv3d(intermediate_channels + 3, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.intermediate_channels = intermediate_channels

        assert downsampling_factor >= 1
        if downsampling_factor == 1:
            self.pooler = nn.Identity()
        else:
            ksize = downsampling_factor + 1
            ksize = (1, ksize, ksize)

            stride = downsampling_factor
            stride = (1, stride, stride)

            padding = int(math.floor((downsampling_factor + 1) / 2))
            padding = (0, padding, padding)
            self.pooler = nn.AvgPool3d(kernel_size=ksize, stride=stride, padding=padding)

    @staticmethod
    def create_spatiotemporal_grid(height, width, time, t_scale, dtype=torch.float32, device="cpu"):
        x = (torch.arange(width)).float().cuda() / ((width - 1) * 0.25) - 2
        y = (torch.arange(height)).float().cuda() / ((height - 1) * 0.5) - 1
        t = ((torch.arange(time)).float().cuda() / ((time - 1) * 0.5) - 1) * t_scale
        return torch.stack(torch.meshgrid(t, y, x), dim=0)  # [3, T, H, W]

    def forward(self, x):
        """
        :param x: tensor of shape [N, C, T, H, W]
        :return: tensor of shape [N, C, T, H, W]
        """
        N, C, T, H, W = x.shape
        theta_x = self.theta(x).permute(0, 2, 3, 4, 1).reshape(N, T * H * W, self.intermediate_channels)
        phi_x = self.pooler(self.phi(x))

        # print("x: ", x.shape)
        Hd, Wd = phi_x.shape[-2:]
        phi_x = phi_x.view(N, self.intermediate_channels, T * Hd * Wd)
        # print("phi_x: ", phi_x.shape)

        f_x = torch.matmul(theta_x, phi_x)  # [N, T*H*W, T*Hd*Wd]
        f_x = f_x.softmax(dim=-1)
        # print("f_x: ", f_x.shape)

        g_x = self.g(x)
        g_x = self.pooler(g_x)  # [N, C, T, Hd, Wd]
        # print("g_x: ", g_x.shape)

        # append coordinate grid to g
        grid = self.create_spatiotemporal_grid(
            Hd, Wd, T, 0.1, x.dtype, x.device).unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, 1, 1, 1]
        g_x = torch.cat((g_x, grid.detach()), dim=1)  # [N, C+3, T, Hd, Wd]

        g_x = g_x.permute(0, 2, 3, 4, 1).reshape(N, T * Hd * Wd, self.intermediate_channels + 3)

        y = torch.matmul(f_x, g_x)  # [N, T*H*W, C]
        y = y.permute(0, 2, 1).reshape(N, self.intermediate_channels + 3, T, H, W)

        return self.W(y)


class NonlocalOffsetEmbeddingHead(nn.Module):
    def __init__(self, in_channels, nonlocal_inter_channels, embedding_size, downsampling_factor, add_spatial_coord=True):
        super(self.__class__, self).__init__()

        self.nonlocal_block = NonLocalBlock3DWithDownsampling(in_channels, nonlocal_inter_channels, downsampling_factor)
        self.conv_offset = nn.Conv3d(in_channels + 3, embedding_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.embedding_size = embedding_size
        self.add_spatial_coord = add_spatial_coord

    def forward(self, x):
        """
        :param x: tensor of shape [N, C, T, H, W]
        :return: embedding map of shape [N, E, T, H, W]
        """
        # assert len(x) == 1
        # x = x[0]
        N, C, T, H, W = x.shape

        # comment: t_scale = 1/(H/T): eg:-352/8
        grid = self.nonlocal_block.create_spatiotemporal_grid(
            H, W, T, 0.1, x.dtype, x.device).unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, 1, 1, 1]
        zeros = torch.zeros(N, C-3, T, H, W, dtype=x.dtype, device=x.device)
        grid_cat = torch.cat((grid, zeros), dim=1)

        x = x + grid_cat.detach()

        x = x + self.nonlocal_block(x)
        x = torch.cat((x, grid), dim=1)

        x = self.conv_offset(x)  # [N, 2, T, H, W]

        # grid = grid[:, 1:]
        if self.embedding_size > 3:
            zeros = torch.zeros(N, self.embedding_size - 3, T, H, W, dtype=x.dtype, device=x.device)
            grid_cat = torch.cat((grid, zeros), dim=1)
        elif self.embedding_size == 3:
            grid_cat = grid
        else:  # embedding size == 1
            grid_cat = torch.tensor(0, dtype=x.dtype, device=x.device)

        return x + grid_cat.detach() if self.add_spatial_coord else x
