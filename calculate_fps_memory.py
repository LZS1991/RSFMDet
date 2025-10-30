import time
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
import psutil
import os
from tqdm import tqdm

# 定义常见图片格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

def calculate_fps(model, dataset_name, warmup_iter=50, test_iter=50):
    """
    计算模型的FPS

    Args:
        model: 加载好的检测模型
        img: 测试图像
        warmup_iter: 热身迭代次数，用于稳定测量
        test_iter: 测试迭代次数，用于计算平均值

    Returns:
        fps: 每秒处理的帧数
        avg_time: 平均推理时间(秒)
    """
    # 热身，排除初始的不稳定状态

    image_files = []
    for filename in os.listdir("./infer_data/{}".format(dataset_name)):
        file_path = os.path.join("./infer_data/{}".format(dataset_name), filename)
        # 只保留文件，且扩展名属于图片格式
        if os.path.isfile(file_path) and filename.lower().endswith(IMAGE_EXTENSIONS):
            image_files.append(file_path)

    # 开始计时测试
    start_time = time.time()
    for i in tqdm(range(test_iter), desc="测试FPS"):
        # 获取dataset_name目录下的第i张图片，但是图片名称不是以i命名
        image_file = image_files[i]
        inference_detector(model, image_file)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / test_iter
    fps = test_iter / total_time

    return fps, avg_time


def calculate_memory_usage(model, img, use_gpu=True):
    """
    计算模型的内存占用

    Args:
        model: 加载好的检测模型
        img: 测试图像
        use_gpu: 是否使用GPU

    Returns:
        memory_usage: 内存使用情况字典
    """
    memory_usage = {}

    # 测量CPU内存使用
    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # 执行一次推理
    inference_detector(model, img)

    cpu_mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage['cpu_memory_mb'] = cpu_mem_after - cpu_mem_before

    # 测量GPU内存使用
    if use_gpu and torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        # 再执行一次推理以测量GPU内存
        inference_detector(model, img)
        gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_usage['gpu_memory_allocated_mb'] = gpu_mem_after - gpu_mem_before

        # 缓存内存
        gpu_cache = torch.cuda.memory_cached() / 1024 / 1024  # MB
        memory_usage['gpu_memory_cached_mb'] = gpu_cache

    return memory_usage


def main(config_file, checkpoint_file, test_image, use_gpu=True, dataset_name=None, model_config_name=None):
    """
    主函数：加载模型并计算FPS和内存占用

    Args:
        config_file: 模型配置文件路径
        checkpoint_file: 模型权重文件路径
        test_image: 测试图像路径
        use_gpu: 是否使用GPU
    """
    # 设置设备
    device = 'cuda:0' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    print(f"使用设备: {device}")

    # 初始化模型，并预热处理一下
    print("加载模型中...")
    model = init_detector(config_file, checkpoint_file, device=device)
    _, _ = calculate_fps(model, dataset_name)

    # 计算FPS
    print("开始计算FPS...")
    fps, avg_time = calculate_fps(model, dataset_name)
    print(f"FPS: {fps:.2f}")
    print(f"{model_config_dir}平均推理时间: {avg_time:.4f}秒")

    # 计算内存占用
    print("开始计算内存占用...")
    memory_usage = calculate_memory_usage(model, test_image, use_gpu)

    print("\n内存占用情况:")
    print(f"CPU内存增加: {memory_usage['cpu_memory_mb']:.2f} MB")
    if 'gpu_memory_allocated_mb' in memory_usage:
        print(f"GPU内存分配: {memory_usage['gpu_memory_allocated_mb']:.2f} MB")
        print(f"GPU缓存内存: {memory_usage['gpu_memory_cached_mb']:.2f} MB")

    # 释放模型和显存占用
    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    del model


if __name__ == "__main__":
    # 请根据你的实际情况修改以下路径
    dataset_name = "xwheel"
    # dataset_name = "ucas"
    # dataset_name = "dota"
    # dataset_name = "dior"

    model_config_dir = [
        "retinanet_convnextv2_atto_fpn_3x_coco",
        "retinanet_pvtv2-b0_fpn_3x_coco",
        "retinanet_pslt_fpn_3x_coco",
        "mask_rcnn_convnextv2_atto_fpn_3x_coco",
        "mask_rcnn_pvtv2-b0_fpn_3x_coco",
        "mask_rcnn_pslt_fpn_3x_coco",
    ]

    model_weight_names = [
        "epoch_50.pth",
        "epoch_50.pth",
        "epoch_50.pth",
        "epoch_50.pth",
        "epoch_50.pth",
        "epoch_50.pth",
    ]

    for i in range(len(model_config_dir)):
        model_config_name = model_config_dir[i]
        model_weight_name = model_weight_names[i]

        config_file = './tools_{}/work_dirs/{}/{}.py'.format(dataset_name, model_config_name, model_config_name)  # 模型配置文件路径
        checkpoint_file = './tools_{}/work_dirs/{}/{}'.format(dataset_name, model_config_name, model_weight_name)  # 模型权重文件路径
        if dataset_name == "dota" or dataset_name == "ucas":
            test_image = './infer_data/{}.png'.format(dataset_name)  # 测试图像路径
        else:
            test_image = './infer_data/{}.jpg'.format(dataset_name)  # 测试图像路径
        use_gpu = True  # 是否使用GPU

        main(config_file, checkpoint_file, test_image, use_gpu, dataset_name, model_config_name)