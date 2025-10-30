# mmdetection/mmdet/models/backbones/dinov3.py
import os
import sys

import torch.nn as nn
import torch

from mmdet.models import BACKBONES

os.environ['TORCH_HOME'] = "../models"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.chdir(sys.path[0])

root_dir = "E:/08-PythonProject/20221214 EfficientViT/EfficientViTMMDetection/"
repo_dir_default = root_dir + "./dinov3-main"  # 这里写的是dinov3的源码位置
model_default = "dinov3_vitl16"  # 这里是指定用哪个模型，具体参考dinov3的readme。
weight_path_default = root_dir + "./dinov3-main/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"  # 这里是模型的权重文件路径


# @BACKBONES.register_module()
class DinoV3ViT(nn.Module):

    def __init__(
            self,
            repo_dir=repo_dir_default,
            model=model_default,
            weight_path=weight_path_default,
            out_indices=[0, 1, 2, 3],
            freeze=True
    ):
        super(DinoV3ViT, self).__init__()
        self.out_indices = out_indices
        self.dinov3_vitl16 = torch.hub.load(
            repo_dir,
            model,
            source='local',
            pretrained=False,
            weights=weight_path
        )

        if freeze:
            for param in self.dinov3_vitl16.parameters():
                param.requires_grad = False

    def forward(self, x):  # should return a tuple
        outs = self.dinov3_vitl16.get_intermediate_layers(
            x,
            self.out_indices,
            reshape=True
        )
        return tuple(outs)


if __name__ == "__main__":
    # dinov3 = DinoV3ConvNeXt()     # 默认base模型
    dinov3 = DinoV3ViT(  # 可以修改参数使用不同的模型
        model=model_default,  # 指定模型
        weight_path=weight_path_default  # 指定权重文件路径
    )
    # 打印每个特征图的shape，后续配置文件修改neck里的参数需要根据这个输出来。
    print([i.shape for i in dinov3(torch.randn(1, 3, 512, 512))])
