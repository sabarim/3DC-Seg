import torch
import torch.nn as nn
import torch.nn.functional as F

from network.embedding_head import NonLocalBlock3DWithDownsampling


class MultiscaleCombinedHeadLongTemporalWindow(nn.Module):
    def __init__(self, in_channels, num_classes, variance_output, variance_per_axis, **kwargs):
        super().__init__()

        self.embedding_size = 3
        self.variance_channels = (self.embedding_size if variance_per_axis else 1) if variance_output else 0
        self.seed_map = kwargs.get("seed_map", False)
        nonlocal_inter_channels = kwargs.get("nonlocal_inter_channels", 128)
        conv_inter_channels = kwargs.get("conv_inter_channels", 128)
        self.add_spatial_coord = kwargs.get("add_spatial_coord", False)
        if not self.add_spatial_coord:
            print("Spatial coordinates are not added to the feature maps in the embedding head")

        # 32/8 (spatial/temporal stride) branch
        nl_in_channels = in_channels + 3 if self.add_spatial_coord else in_channels
        self.nonlocal_32x = NonLocalBlock3DWithDownsampling(nl_in_channels, nonlocal_inter_channels, 1, in_channels)
        self.conv_32x_1 = nn.Conv3d(in_channels, conv_inter_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv_32x_2 = nn.Conv3d(conv_inter_channels, conv_inter_channels, kernel_size=3, padding=1)

        # 16/4 branch
        self.nonlocal_16x = NonLocalBlock3DWithDownsampling(nl_in_channels, nonlocal_inter_channels, 1, in_channels)
        self.conv_16x_1 = nn.Conv3d(in_channels, conv_inter_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # 8/2 branch
        self.conv_8x_1 = nn.Conv3d(in_channels, conv_inter_channels, kernel_size=3, padding=(1, 2, 2), dilation=(1, 2, 2))

        # 4/1 branch
        self.conv_4x_1 = nn.Conv3d(in_channels, conv_inter_channels, kernel_size=3, padding=1)

        # semseg branch (after combining multi-scale maps)
        self.conv_semseg = nn.Conv3d(conv_inter_channels * 4, conv_inter_channels, kernel_size=3, padding=1)
        self.conv_semseg_out = nn.Conv3d(conv_inter_channels, num_classes, kernel_size=1, padding=0, bias=False)

        # embedding branch (after combining multi-scale maps)
        self.conv_embedding = nn.Conv3d(
            conv_inter_channels * 4, conv_inter_channels, kernel_size=3, padding=1)
        self.conv_embedding_out = nn.Conv3d(
            conv_inter_channels, self.embedding_size, kernel_size=1, padding=0, bias=False)

        if self.variance_channels > 0:
            self.conv_variance_out = nn.Conv3d(
                conv_inter_channels, self.variance_channels, kernel_size=1, padding=0, bias=True)
        if self.seed_map:
            self.conv_seed_out = nn.Conv3d(
                conv_inter_channels, 1, kernel_size=1, padding=0, bias=True)

        self.register_buffer("time_scale", torch.tensor(kwargs.get('time_scale', 0.2), dtype=torch.float32))
        self.register_buffer("tanh_premultiplier", torch.tensor(0.25, dtype=torch.float32))

    def forward_32_8(self, x):
        N, C, T, H, W = x.shape

        grid = self.nonlocal_32x.create_spatiotemporal_grid(
            H, W, T, self.time_scale, x.dtype, x.device).unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, 1, 1, 1]
        t = torch.cat((x, grid.detach()), dim=1) if self.add_spatial_coord else x
        x = x + self.nonlocal_32x(t)

        # 32/8 -> 16/4
        x = F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)
        x = F.relu(self.conv_32x_1(x))

        # 16/4 -> 8/2
        x = F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)
        x = F.relu(self.conv_32x_2(x))

        # 8/2 -> 4/1
        return F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)

    def forward_16_4(self, x):
        N, C, T, H, W = x.shape

        grid = self.nonlocal_16x.create_spatiotemporal_grid(
            H, W, T, self.time_scale, x.dtype, x.device).unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, 1, 1, 1]
        t = torch.cat((x, grid.detach()), dim=1) if self.add_spatial_coord else x
        x = x + self.nonlocal_16x(t)

        # 16/4 -> 8/2
        x = F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)
        x = F.relu(self.conv_16x_1(x))

        # 8/2 -> 4/1
        return F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)

    def forward_8_2(self, x):
        x = F.relu(self.conv_8x_1(x))

        # 8/2 -> 4/1
        return F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)

    def forward_4_1(self, x):
        return F.relu(self.conv_4x_1(x))

    def semseg_branch(self, x):
        x = F.relu(self.conv_semseg(x))
        return self.conv_semseg_out(x)

    def embedding_branch(self, x):
        x = F.relu(self.conv_embedding(x))

        N, C, T, H, W = x.shape
        grid = self.nonlocal_32x.create_spatiotemporal_grid(
            H, W, T, self.time_scale, x.dtype, x.device).unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, T, H, W]

        embeddings = self.conv_embedding_out(x)
        embeddings = (embeddings * self.tanh_premultiplier).tanh() + grid.detach()

        if self.variance_channels > 0:
            variances = self.conv_variance_out(x)
            embeddings = torch.cat((embeddings, variances), dim=1)
        if self.seed_map:
            seed = self.conv_seed_out(x)
            embeddings = torch.cat((embeddings, seed), dim=1)
        return embeddings

    def forward(self, x):
        """
        :param x: list of multiscale feature map tensors of shape [N, C, T, H, W].
        Order should be [32/8, 16/4, 8/2, 4/1]
        :return: dict with keys 'embeddings', 'variances' and 'semseg' maps of shape [N, E/E/C, T, H, W]
        """
        assert len(x) == 4
        scale_forward_fns = [self.forward_32_8, self.forward_16_4, self.forward_8_2, self.forward_4_1]
        x = [fn(feats) for fn, feats in zip(scale_forward_fns, x)]
        # for a in x:
        #     print(a.shape)

        x = torch.cat(x, dim=1)

        semseg_logits = self.semseg_branch(x)
        embeddings = self.embedding_branch(x)

        return semseg_logits, embeddings


if __name__ == '__main__':
    head = MultiscaleCombinedHeadLongTemporalWindow(256, 2, True, True).cuda()

    feature_maps = [
        torch.zeros(1, 256, 1, 32, 48, dtype=torch.float32, device='cuda'),
        torch.zeros(1, 256, 2, 64, 96, dtype=torch.float32, device='cuda'),
        torch.zeros(1, 256, 4, 128, 192, dtype=torch.float32, device='cuda'),
        torch.zeros(1, 256, 8, 256, 384, dtype=torch.float32, device='cuda')
    ]

    embeddings_out, semseg_logits_out = head(feature_maps)
    print(embeddings_out.shape)
    print(semseg_logits_out.shape)
