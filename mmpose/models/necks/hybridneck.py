import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, ConvTranspose2d
from mmpose.registry import MODELS

# 定义辅助模块

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else nn.SiLU(inplace=True)  # 或根据需要自定义激活函数

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='SiLU'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else nn.SiLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="SiLU"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# 定义 HybridNeck 模块

@MODELS.register_module()
class HybridNeck(nn.Module):
    """
    HybridNeck 结合了特征金字塔网络（FPN）和路径聚合网络（PAN）的优势，
    使用 RepVggBlock 和 CSPRepLayer 进行高效的特征融合。

    Args:
        in_channels (list[int]): 来自骨干网络每个阶段的输入通道数。
        out_channels (int): 输出的通道数。
        num_outs (int): 输出的尺度数。
        expansion (float): CSPRepLayer 中的通道扩展倍数。默认 1.0。
        depth_mult (float): CSPRepLayer 中的块数量乘数。默认 1.0。
        act (str): 激活函数类型。默认 'SiLU'。
        norm_cfg (dict): 归一化层配置。默认 None。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='SiLU',
                 norm_cfg=None):
        super(HybridNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.expansion = expansion
        self.depth_mult = depth_mult
        self.act = act
        self.norm_cfg = norm_cfg

        # 检查 num_outs 是否合理
        if self.num_outs < self.num_ins:
            raise ValueError("num_outs 应该大于或等于 in_channels 的长度")

        # 输入通道投影
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, out_channels, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(out_channels))
            ]))
            self.input_proj.append(proj)

        # FPN 部分的侧向卷积和融合块
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for i in range(self.num_ins - 1, 0, -1):
            # 侧向卷积
            lateral_conv = ConvNormLayer(
                ch_in=out_channels,
                ch_out=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act=act
            )
            self.lateral_convs.append(lateral_conv)

            # FPN 融合块
            fpn_block = CSPRepLayer(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                num_blocks=round(3 * depth_mult),
                expansion=expansion,
                bias=False,
                act=act
            )
            self.fpn_blocks.append(fpn_block)

        # PAN 部分的下采样卷积和融合块
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for i in range(self.num_ins - 1):
            # 下采样卷积
            downsample_conv = ConvNormLayer(
                ch_in=out_channels,
                ch_out=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                act=act
            )
            self.downsample_convs.append(downsample_conv)

            # PAN 融合块
            pan_block = CSPRepLayer(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                num_blocks=round(3 * depth_mult),
                expansion=expansion,
                bias=False,
                act=act
            )
            self.pan_blocks.append(pan_block)

        # 最后的卷积层，调整输出通道数
        self.final_conv = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg
        )

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): 来自骨干网络的特征图列表。
        
        Returns:
            list[Tensor]: 融合后的多尺度特征图列表。
        """
        assert len(inputs) == self.num_ins, \
            f"输入特征图数量 {len(inputs)} 不等于 in_channels 的长度 {self.num_ins}"

        # 输入投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(inputs)]

        # FPN 部分的特征融合（自上而下）
        inner_outs = [proj_feats[-1]]  # 开始于最深层特征
        for idx in range(self.num_ins - 1, 0, -1):
            # 调整当前特征的通道数
            feat_high = self.lateral_convs[self.num_ins - 1 - idx](inner_outs[0])
            # 上采样
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            # 获取对应的低层特征
            feat_low = proj_feats[idx - 1]
            # 拼接
            fused_feat = torch.cat([upsample_feat, feat_low], dim=1)
            # 融合
            fused_feat = self.fpn_blocks[self.num_ins - 1 - idx](fused_feat)
            # 插入到 inner_outs 的最前面
            inner_outs.insert(0, fused_feat)

        # PAN 部分的特征融合（自下而上）
        outs = [inner_outs[0]]
        for idx in range(self.num_ins - 1):
            # 下采样
            downsample_feat = self.downsample_convs[idx](outs[-1])
            # 获取对应的高层融合特征
            feat_high = inner_outs[idx + 1]
            # 拼接
            pan_fused_feat = torch.cat([downsample_feat, feat_high], dim=1)
            # 融合
            pan_fused_feat = self.pan_blocks[idx](pan_fused_feat)
            # 添加到输出列表
            outs.append(pan_fused_feat)

        # 最后的卷积调整
        outs = [self.final_conv(out) for out in outs]

        return outs
